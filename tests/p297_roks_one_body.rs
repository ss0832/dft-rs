// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All nine common-orbital susceptibilities against freshly converged
//! displaced ROKS density matrices on a nonsymmetric four-centre molecule.
//! This validates the general even source, not transverse Ramsey response.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::integrals::london::ComplexMatrix;
use dft_rs::linalg::Matrix;
use dft_rs::response::nuclear::open_shell_operator_response;
use dft_rs::scf::roks::run_roks_with_orbitals;

#[test]
fn common_orbital_general_sources_have_all_nine_displaced_density_fits() {
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
    let controls = Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("reference", "roks")
        .with("charge", 1i32)
        .with("multiplicity", 2usize)
        .with("dispersion", "none")
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-14)
        .with("conv_tol_dp", 1e-14)
        .with("max_iter", 500usize)
        .with("guess", "core");
    let method = MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let (reference, orbitals) = run_roks_with_orbitals(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )
    .unwrap();
    assert!(reference.converged);
    let n = setup.basis.nao_sph;
    let sources = (0..3)
        .map(|axis| {
            let mut h = Matrix::zeros(n, n);
            for i in 0..n {
                for j in i..n {
                    h[(i, j)] = 0.13 * ((i + 3 * j + 7 * axis + 1) as f64 * 0.37).sin();
                    h[(j, i)] = h[(i, j)];
                }
            }
            let h = ComplexMatrix {
                re: h,
                im: Matrix::zeros(n, n),
            };
            [h.clone(), h]
        })
        .collect::<Vec<_>>();
    let config = dft_rs::response::ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let value = open_shell_operator_response(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference,
        &orbitals,
        &setup.options,
        &sources,
        dft_rs::response::Frequency::Static,
        &config,
    )
    .unwrap();
    assert!(value.residuals.iter().all(|r| *r < 1e-12));
    assert!(value
        .susceptibility
        .iter()
        .flatten()
        .all(|v| v.im.abs() < 1e-14));
    let names = (0..9)
        .map(|p| {
            format!(
                "C1 ROHF common-orbital one-body susceptibility ({},{})",
                p / 3,
                p % 3
            )
        })
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[
            [0.08, 0.06, 0.04, 0.03],
            [0.02, 0.015, 0.01, 0.0075],
            [0.2, 0.15, 0.1, 0.075],
        ],
        |steps| {
            let mut errors = vec![[0.0; 4]; 9];
            for (k, &h) in steps.iter().enumerate() {
                for b in 0..3 {
                    let at = |t: f64| {
                        let mut options = setup.options.clone();
                        let mut source = sources[b][0].re.clone();
                        for x in source.as_mut_slice() {
                            *x *= t;
                        }
                        options.extra_hcore = Some(source);
                        let (scf, _) = run_roks_with_orbitals(
                            &setup.molecule,
                            &setup.basis,
                            &options,
                            &setup.functional,
                            setup.grid.as_ref(),
                        )
                        .unwrap();
                        assert!(scf.converged);
                        (0..3)
                            .map(|a| {
                                (0..2)
                                    .map(|s| {
                                        sources[a][s]
                                            .re
                                            .as_slice()
                                            .iter()
                                            .zip(scf.density[s].as_slice())
                                            .map(|(h, p)| h * p)
                                            .sum::<f64>()
                                    })
                                    .sum::<f64>()
                            })
                            .collect::<Vec<_>>()
                    };
                    let plus = at(h);
                    let minus = at(-h);
                    for a in 0..3 {
                        errors[3 * a + b][k] = ((plus[a] - minus[a]) / (2.0 * h)
                            - value.susceptibility[a][b].re)
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("complete common-orbital one-body susceptibility", &failures);
}

#[test]
fn complex_common_orbital_vertices_match_one_electron_spectral_response() {
    use dft_rs::dfpt::orbitals::{band_density, generalized_orbitals};
    use dft_rs::periodic::BlochMatrix;
    use dft_rs::response::{complex::Complex as C, Frequency};
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
    let controls = Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("reference", "roks")
        .with("charge", 3i32)
        .with("multiplicity", 2usize)
        .with("dispersion", "none")
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-14)
        .with("conv_tol_dp", 1e-14)
        .with("max_iter", 500usize)
        .with("guess", "core");
    let method = MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let (reference, orbitals) = run_roks_with_orbitals(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )
    .unwrap();
    assert!(reference.converged);
    let n = setup.basis.nao_sph;
    let sources = (0..3)
        .map(|axis| {
            let mut h = ComplexMatrix::zeros(n, n);
            for i in 0..n {
                for j in i..n {
                    let phase = (i + 3 * j + 7 * axis + 1) as f64;
                    h.re[(i, j)] = 0.13 * (phase * 0.37).sin();
                    h.re[(j, i)] = h.re[(i, j)];
                    if i != j {
                        h.im[(i, j)] = 0.09 * (phase * 0.29).cos();
                        h.im[(j, i)] = -h.im[(i, j)];
                    }
                }
            }
            [h.clone(), h]
        })
        .collect::<Vec<_>>();
    let overlap = BlochMatrix {
        re: reference.overlap.clone(),
        im: Matrix::zeros(n, n),
    };
    let h = BlochMatrix {
        re: reference.h_core.clone(),
        im: Matrix::zeros(n, n),
    };
    let bands = generalized_orbitals(&h, &overlap, 1e-12).unwrap();
    let c = &bands.coefficients;
    let mo = |source: &ComplexMatrix, i: usize, a: usize| {
        let mut out = C::ZERO;
        for mu in 0..n {
            for nu in 0..n {
                let left = C::new(c.re[(mu, i)], c.im[(mu, i)]).conj();
                let right = C::new(c.re[(nu, a)], c.im[(nu, a)]);
                out += left * C::new(source.re[(mu, nu)], source.im[(mu, nu)]) * right;
            }
        }
        out
    };
    let config = dft_rs::response::ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut static_value = None;
    for frequency in [
        Frequency::Static,
        Frequency::Retarded {
            omega: 0.19,
            damping: 0.037,
        },
        Frequency::Imaginary { frequency: 0.21 },
    ] {
        let report = open_shell_operator_response(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference,
            &orbitals,
            &setup.options,
            &sources,
            frequency,
            &config,
        )
        .unwrap();
        let z = frequency.energy().unwrap();
        let mut defect = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let mut expected = C::ZERO;
                for r in 1..n {
                    let delta = bands.energies[r] - bands.energies[0];
                    let x = mo(&sources[a][0], 0, r);
                    let y = mo(&sources[b][0], r, 0);
                    expected +=
                        x * y / (z - C::from(delta)) - (x * y).conj() / (z + C::from(delta));
                }
                defect = defect.max((expected - report.susceptibility[a][b]).abs());
            }
        }
        println!(
            "complex ROKS one-electron frequency={frequency:?} all9 spectral defect={defect:.12e}"
        );
        assert!(defect < 1e-11);
        if frequency == Frequency::Static {
            static_value = Some(report);
        }
    }
    let report = static_value.unwrap();
    let names = (0..9)
        .map(|p| {
            format!(
                "C1 complex ROHF one-electron susceptibility ({},{})",
                p / 3,
                p % 3
            )
        })
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[
            [0.08, 0.06, 0.04, 0.03],
            [0.02, 0.015, 0.01, 0.0075],
            [0.2, 0.15, 0.1, 0.075],
        ],
        |steps| {
            let mut errors = vec![[0.0; 4]; 9];
            for (k, &step) in steps.iter().enumerate() {
                for b in 0..3 {
                    let at = |t: f64| {
                        let mut op = h.clone();
                        op.add_scaled(
                            &BlochMatrix {
                                re: sources[b][0].re.clone(),
                                im: sources[b][0].im.clone(),
                            },
                            t,
                        );
                        let orbitals = generalized_orbitals(&op, &overlap, 1e-12).unwrap();
                        let density = band_density(&orbitals, &[1.0, 0.0, 0.0, 0.0]).unwrap();
                        (0..3)
                            .map(|a| {
                                BlochMatrix {
                                    re: sources[a][0].re.clone(),
                                    im: sources[a][0].im.clone(),
                                }
                                .trace_product(&density)
                            })
                            .collect::<Vec<_>>()
                    };
                    let plus = at(step);
                    let minus = at(-step);
                    for a in 0..3 {
                        errors[3 * a + b][k] = ((plus[a] - minus[a]) / (2.0 * step)
                            - report.susceptibility[a][b].re)
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "complete complex common-orbital one-electron susceptibility",
        &failures,
    );
}
