// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every shielding component on C1 common-orbital and complex-spin manifolds.
//! Independent finite-field SCF expectations validate analytic response.
//! Spectral tests also exercise the same full-spin UKS coupling route as
//! standalone Ramsey tensors, rather than the restricted-only legacy kernel.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/complex_common_hf.rs"]
mod common_oracle;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::math::Vec3;
use dft_rs::periodic::{
    spinor_scf::{run_spinor_scf, spin_diagonal, SpinorScfConfig},
    BlochMatrix,
};
use dft_rs::properties::{
    magnetic::{common_orbital_nmr_response, spinor_nmr_response, Gauge},
    nmr::{angular_momentum_operator, paramagnetic_operator, shielding_mixed_operator},
};
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::response::ResponseSolverConfig;

fn system() -> System {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    );
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    system
}
fn parameters() -> Value {
    Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("dispersion", "none")
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-13)
        .with("conv_tol_grad", 1e-12)
        .with("max_iter", 500usize)
        .with("response_tolerance", 1e-11)
}
fn stages() -> [([f64; 4], stencils::Stencil); 4] {
    [
        ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
        ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
        ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
    ]
}

#[test]
fn all_common_orbital_shieldings_fit_independent_complex_constrained_scf() {
    let system = system();
    let parameters = parameters()
        .with("multiplicity", 3usize)
        .with("guess", "core");
    let method =
        MolecularMethod::from_controls(&Controls::from_value(&parameters).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let (reference, common) = dft_rs::scf::roks::run_roks_with_orbitals(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )
    .unwrap();
    let origin = Vec3::new(0.19, -0.11, 0.23);
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let response = common_orbital_nmr_response(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference,
        &common,
        &setup.options,
        Gauge::CommonOrigin,
        origin,
        &config,
    )
    .unwrap();
    let eri = dft_rs::integrals::eri_incore(
        &setup.basis,
        dft_rs::integrals::CoulombKernel::Coulomb,
        1280,
    )
    .unwrap();
    let model = common_oracle::Model::new(
        reference.overlap.clone(),
        reference.h_core.clone(),
        eri,
        reference.e_nuc,
        [reference.n_alpha, reference.n_beta],
    );
    let fields = angular_momentum_operator(&setup.basis, origin).map(|mut m| {
        for v in m.as_mut_slice() {
            *v *= -0.5;
        }
        m
    });
    let zero = model.stationary(&common, &fields[0], 0.0);
    assert!((zero.energy - reference.energy).abs() < 1e-10);
    assert!(zero.residual < 1e-11);
    let nuclear: Vec<_> = setup
        .molecule
        .atoms
        .iter()
        .map(|a| paramagnetic_operator(&setup.basis, a.position))
        .collect();
    let mixed: Vec<_> = setup
        .molecule
        .atoms
        .iter()
        .map(|a| shielding_mixed_operator(&setup.basis, a.position, origin, false))
        .collect();
    let scale = dft_rs::constants::FINE_STRUCTURE.powi(2) * 1e6;
    let mut failures = Vec::new();
    for field in 0..3 {
        let names = (0..12)
            .map(|k| format!("C1 ROHF shielding n={} moment={} B={field}", k / 3, k % 3))
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &names,
            &stages(),
            |steps, stencil| {
                let at = |t| {
                    let state = model.stationary(&common, &fields[field], t);
                    assert!(state.residual < 1e-11);
                    (0..12)
                        .map(|k| {
                            let (n, a) = (k / 3, k % 3);
                            state
                                .density
                                .iter()
                                .map(|p| {
                                    -p.im.frobenius_dot(&nuclear[n][a])
                                        + t * p.re.frobenius_dot(&mixed[n][field][a])
                                })
                                .sum::<f64>()
                                * scale
                        })
                        .collect::<Vec<_>>()
                };
                let mut errors = vec![[0.0; 4]; 12];
                for (step, &h) in steps.iter().enumerate() {
                    let (p, m) = (at(h), at(-h));
                    let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        (vec![0.0; 12], vec![0.0; 12])
                    };
                    for k in 0..12 {
                        errors[k][step] = (stencil.first(p[k], m[k], p2[k], m2[k], h)
                            - response.value[k / 3].tensor[k % 3][field])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    println!(
        "ROHF NMR raw response={:?} stationarity={:.12e}",
        response.residuals, response.reference_stationarity
    );
    components::assert_no_failures("all 36 common-orbital shielding components", &failures);
}

#[test]
fn all_spinor_shieldings_fit_independent_complex_scf() {
    let system = system();
    let parameters = parameters()
        .with("charge", 1)
        .with("multiplicity", 2usize)
        .with("unrestricted", true);
    let method =
        MolecularMethod::from_controls(&Controls::from_value(&parameters).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    let base = spinor_operator::source(setup.basis.nao_sph);
    let scf_config = SpinorScfConfig {
        energy_tolerance: 1e-14,
        density_tolerance: 1e-14,
        orbital_tolerance: 1e-14,
        ..Default::default()
    };
    let solve = |operator| {
        let backend = MolecularSpinorBackend::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &seed.scf,
            &setup.options,
            operator,
            [0.3, 0.4, 0.5],
            MolecularSpinorFunctionalModel::SpinCovariant,
            1e-12,
        )
        .unwrap();
        let state = run_spinor_scf(&backend, &scf_config).unwrap();
        assert!(state.converged, "SCF residual {}", state.orbital_residual);
        state
    };
    let reference = solve(base.clone());
    let origin = Vec3::new(0.19, -0.11, 0.23);
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let response = spinor_nmr_response(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference,
        &setup.options,
        &base,
        &[],
        Gauge::CommonOrigin,
        origin,
        &config,
    )
    .unwrap();
    let fields = angular_momentum_operator(&setup.basis, origin).map(|mut m| {
        for v in m.as_mut_slice() {
            *v *= -0.5;
        }
        spin_diagonal(&BlochMatrix {
            re: dft_rs::linalg::Matrix::zeros(m.rows, m.cols),
            im: m,
        })
    });
    let nuclear: Vec<_> = setup
        .molecule
        .atoms
        .iter()
        .map(|a| {
            paramagnetic_operator(&setup.basis, a.position).map(|mut m| {
                for v in m.as_mut_slice() {
                    *v *= -1.0;
                }
                spin_diagonal(&BlochMatrix {
                    re: dft_rs::linalg::Matrix::zeros(m.rows, m.cols),
                    im: m,
                })
            })
        })
        .collect();
    let mixed: Vec<_> = setup
        .molecule
        .atoms
        .iter()
        .map(|a| {
            shielding_mixed_operator(&setup.basis, a.position, origin, false).map(|row| {
                row.map(|m| {
                    spin_diagonal(&BlochMatrix {
                        im: dft_rs::linalg::Matrix::zeros(m.rows, m.cols),
                        re: m,
                    })
                })
            })
        })
        .collect();
    let scale = dft_rs::constants::FINE_STRUCTURE.powi(2) * 1e6;
    let mut failures = Vec::new();
    for field in 0..3 {
        let names = (0..12)
            .map(|k| format!("C1 GHF shielding n={} moment={} B={field}", k / 3, k % 3))
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &names,
            &stages(),
            |steps, stencil| {
                let at = |t| {
                    let mut operator = base.clone();
                    operator.add_scaled(&fields[field], t);
                    let state = solve(operator);
                    (0..12)
                        .map(|k| {
                            let (n, a) = (k / 3, k % 3);
                            scale
                                * (nuclear[n][a].trace_product(&state.density[0])
                                    + t * mixed[n][field][a].trace_product(&state.density[0]))
                        })
                        .collect::<Vec<_>>()
                };
                let mut errors = vec![[0.0; 4]; 12];
                for (step, &h) in steps.iter().enumerate() {
                    let (p, m) = (at(h), at(-h));
                    let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        (vec![0.0; 12], vec![0.0; 12])
                    };
                    for k in 0..12 {
                        errors[k][step] = (stencil.first(p[k], m[k], p2[k], m2[k], h)
                            - response.value[k / 3].tensor[k % 3][field])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    println!(
        "GHF NMR raw response={:?} stationarity={:.12e}",
        response.residuals, response.reference_stationarity
    );
    components::assert_no_failures("all 36 generalized shielding components", &failures);
}

#[test]
fn closed_shell_spectra_preserve_reference_specific_routes() {
    let system = system();
    let base = parameters()
        .with("gauge", "common-origin")
        .with("nmr_reference_elements", vec![1usize])
        .with("nmr_reference_shieldings", vec![30.0])
        .with("nmr_couplings", true)
        .with("nmr_spin_model", "exact")
        .with("nuclear_g", vec![5.5856946893; 4])
        .with("nmr_points", 101usize);
    let mut reference: Option<Value> = None;
    for name in ["rks", "uks", "roks", "gks"] {
        let controls = Controls::from_value(&base.clone().with("reference", name)).unwrap();
        let actual = dft_rs::driver::run("nmr-spectrum", &system, &controls).unwrap();
        assert_eq!(actual.get("reference_type").unwrap().as_str(), Some(name));
        assert_eq!(actual.get("converged").unwrap().as_bool(), Some(true));
        assert!(
            actual
                .get("coupling_response_residual")
                .unwrap()
                .as_f64()
                .unwrap()
                < 1e-11
        );
        if let Some(expected) = &reference {
            for key in [
                "shielding_tensors",
                "coupling_constants_hz",
                "spectrum_intensity",
            ] {
                let (_, a) = actual.get(key).unwrap().numeric_array().unwrap();
                let (_, b) = expected.get(key).unwrap().numeric_array().unwrap();
                assert_eq!(a.len(), b.len());
                let defect = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f64, f64::max);
                println!("closed limit reference={name} {key} defect={defect:.12e}");
                assert!(defect < 1e-7, "{name} {key}: {defect}");
            }
        } else {
            reference = Some(actual);
        }
    }
}

#[test]
fn all_london_common_orbital_shieldings_fit_finite_field_scf_in_exact_limits() {
    // With no open class or only one occupied orbital the independent
    // complex SCF has exactly the same common-orbital variational manifold.
    // Interacting open classes have their separate dense constrained test.
    for one_electron in [false, true] {
        let mut system = system();
        for axis in 0..3 {
            let center = system.positions.iter().map(|r| r[axis]).sum::<f64>() / 4.0;
            for r in &mut system.positions {
                r[axis] -= center;
            }
        }
        let mut request = parameters().with("guess", "core");
        if one_electron {
            request.set("charge", 3);
            request.set("multiplicity", 2usize);
        }
        let method =
            MolecularMethod::from_controls(&Controls::from_value(&request).unwrap()).unwrap();
        let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
        let (reference, common) = dft_rs::scf::roks::run_roks_with_orbitals(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )
        .unwrap();
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let response = common_orbital_nmr_response(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference,
            &common,
            &setup.options,
            Gauge::Giao,
            Vec3::zero(),
            &config,
        )
        .unwrap();
        let shifted = common_orbital_nmr_response(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference,
            &common,
            &setup.options,
            Gauge::Giao,
            Vec3::new(0.23, -0.41, 0.37),
            &config,
        )
        .unwrap();
        for (a, b) in response.value.iter().zip(&shifted.value) {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((a.tensor[i][j] - b.tensor[i][j]).abs() < 1e-8);
                }
            }
        }
        let nuclear: Vec<_> = setup
            .molecule
            .atoms
            .iter()
            .map(|a| paramagnetic_operator(&setup.basis, a.position))
            .collect();
        let mixed: Vec<_> = setup
            .molecule
            .atoms
            .iter()
            .map(|a| shielding_mixed_operator(&setup.basis, a.position, Vec3::zero(), true))
            .collect();
        let scale = dft_rs::constants::FINE_STRUCTURE.powi(2) * 1e6;
        let mut failures = Vec::new();
        for axis in 0..3 {
            let names = (0..12)
                .map(|k| {
                    format!(
                        "London ROKS one-electron={one_electron} n={} m={} B={axis}",
                        k / 3,
                        k % 3
                    )
                })
                .collect::<Vec<_>>();
            stencils::fit_components_staged(
                &names,
                &{
                    let mut intervals = stages().to_vec();
                    intervals.push((
                        [0.001, 0.0005, 0.00025, 0.000125],
                        stencils::Stencil::Second,
                    ));
                    intervals
                },
                |steps, stencil| {
                    let at = |t| -> dft_rs::Result<Vec<f64>> {
                        let mut field = [0.0; 3];
                        field[axis] = t;
                        let n = setup.basis.nao_sph;
                        let zero = dft_rs::integrals::london::ComplexMatrix::zeros(n, n);
                        let state = dft_rs::scf::magnetic::magnetic_field_scf_with_core(
                            &setup.molecule,
                            &setup.basis,
                            &setup.functional,
                            setup.grid.as_ref(),
                            &reference,
                            &setup.options,
                            &dft_rs::scf::magnetic::MagneticFieldOptions {
                                field,
                                spin_zeeman: false,
                                series_order: Some(8),
                                series_tolerance: 1e-14,
                            },
                            &[zero.clone(), zero],
                        )?;
                        assert!(
                            state.converged,
                            "finite London SCF residual {}",
                            state.residual
                        );
                        Ok((0..12)
                            .map(|k| {
                                let (n, a) = (k / 3, k % 3);
                                state
                                    .density
                                    .iter()
                                    .map(|p| {
                                        -p.im.frobenius_dot(&nuclear[n][a])
                                            + t * p.re.frobenius_dot(&mixed[n][axis][a])
                                    })
                                    .sum::<f64>()
                                    * scale
                            })
                            .collect::<Vec<_>>())
                    };
                    let mut errors = vec![[0.0; 4]; 12];
                    for (step, &h) in steps.iter().enumerate() {
                        let sample = |t| match at(t) {
                            Ok(values) => values,
                            Err(error) => {
                                eprintln!(
                                    "London finite-field failure B={axis} h={t:.12e}: {error}"
                                );
                                vec![f64::NAN; 12]
                            }
                        };
                        let (p, m) = (sample(h), sample(-h));
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (sample(2.0 * h), sample(-2.0 * h))
                        } else {
                            (vec![0.0; 12], vec![0.0; 12])
                        };
                        for k in 0..12 {
                            errors[k][step] = (stencil.first(p[k], m[k], p2[k], m2[k], h)
                                - response.value[k / 3].tensor[k % 3][axis])
                                .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
        components::assert_no_failures("all 36 London common-orbital limit components", &failures);
    }
}
