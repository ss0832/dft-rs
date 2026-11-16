// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All molecular pair-wavefunction AAT entries against independent displaced
//! real/complex SCF determinants and explicit Slater cross-overlap sums.
//! Common-origin finite fields use explicit one-electron B/B^2 operators;
//! their field-free AO basis needs no London projection approximation.
#[path = "support/complex_pt2.rs"]
mod complex_pt2;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/slater.rs"]
mod slater;
use dft_rs::basis::MolBasis;
use dft_rs::driver::method::{
    molecule_from, prepare_molecular, run_molecular_reference, MolecularMethod, MolecularSetup,
};
use dft_rs::integrals::london::{one_electron_series, ComplexMatrix, FieldPath};
use dft_rs::integrals::{eri_incore, overlap_matrix, CoulombKernel};
use dft_rs::properties::correlated_axial::correlated_molecular_axial_response;
use dft_rs::response::complex::Complex;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::magnetic::{magnetic_field_scf_with_core, MagneticFieldOptions};
use dft_rs::xc::double_hybrid::Pt2Coefficients;
use dft_rs::{Matrix, Vec3};
use std::collections::HashMap;

struct Wavefunction {
    positions: Vec<[f64; 3]>,
    orbitals: [ComplexMatrix; 2],
    amplitudes: [Vec<Complex>; 3],
    phase: Complex,
}
fn complex(c: &Matrix) -> ComplexMatrix {
    ComplexMatrix {
        re: c.clone(),
        im: Matrix::zeros(c.rows, c.cols),
    }
}
fn cross(setup: &MolecularSetup, left: &Wavefunction, right: &Wavefunction) -> Vec<Vec<Complex>> {
    let n = setup.basis.nao_sph;
    let z: Vec<_> = setup.molecule.atoms.iter().map(|a| a.z).collect();
    let both = molecule_from(
        &[z.clone(), z].concat(),
        &[left.positions.clone(), right.positions.clone()].concat(),
        0,
        1,
    )
    .unwrap();
    let mb = MolBasis::build(&both, &setup.basis_set).unwrap();
    let overlap = overlap_matrix(&mb);
    let mut s = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            s[(i, j)] = overlap[(i, n + j)];
        }
    }
    let mut result = vec![vec![Complex::ZERO; 2 * n]; 2 * n];
    for spin in 0..2 {
        for i in 0..n {
            for j in 0..n {
                let mut value = Complex::ZERO;
                for mu in 0..n {
                    for nu in 0..n {
                        value += Complex::new(
                            left.orbitals[spin].re[(mu, i)],
                            -left.orbitals[spin].im[(mu, i)],
                        ) * Complex::new(
                            right.orbitals[spin].re[(nu, j)],
                            right.orbitals[spin].im[(nu, j)],
                        ) * s[(mu, nu)];
                    }
                }
                result[spin * n + i][spin * n + j] = value;
            }
        }
    }
    result
}
fn align(
    setup: &MolecularSetup,
    reference: &Wavefunction,
    state: &mut Wavefunction,
    occupied: [usize; 2],
) {
    let n = setup.basis.nao_sph;
    let determinant: Vec<_> = (0..occupied[0])
        .chain((0..occupied[1]).map(|i| n + i))
        .collect();
    let reference_det = vec![(determinant, Complex::ONE)];
    let overlap = slater::slater_overlap(
        &reference_det,
        &reference_det,
        &cross(setup, reference, state),
    );
    assert!(
        overlap.abs() > 0.5,
        "finite reference changed determinant branch"
    );
    state.phase = overlap.conj() / overlap.abs();
}
fn determinants(
    state: &Wavefunction,
    n: usize,
    o: [usize; 2],
    channel: usize,
    w: Pt2Coefficients,
) -> Vec<(Vec<usize>, Complex)> {
    let zero = state
        .amplitudes
        .each_ref()
        .map(|v| vec![Complex::ZERO; v.len()]);
    let weights = if channel == 1 {
        Pt2Coefficients {
            same_spin: 0.0,
            opposite_spin: 0.0,
        }
    } else {
        w
    };
    let selected = if channel >= 2 {
        Some(channel - 2)
    } else {
        None
    };
    let mut result =
        slater::finite_state([n, n], o, &state.amplitudes, &zero, 0.0, selected, weights);
    for (_, value) in &mut result {
        *value = *value * state.phase;
    }
    if channel == 0 {
        let norm = result
            .iter()
            .map(|(_, z)| (z.conj() * *z).re)
            .sum::<f64>()
            .sqrt();
        for (_, value) in &mut result {
            *value = *value / norm;
        }
    }
    result
}

#[test]
fn every_molecular_rks_uks_pair_axial_entry_has_a_slater_overlap_error_fit() {
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
            for a in 0..3 {
                let mean = positions.iter().map(|r| r[a]).sum::<f64>() / positions.len() as f64;
                for r in &mut positions {
                    r[a] -= mean;
                }
            }
            let method = MolecularMethod {
                xc: Some(xc.into()),
                basis: "sto-3g".into(),
                unrestricted,
                multiplicity: if unrestricted { 2 } else { 1 },
                guess: Some("core".into()),
                max_iter: 300,
                conv_tol: 1e-14,
                conv_tol_dp: 1e-14,
                conv_tol_grad: Some(1e-14),
                grid_level: 0,
                grid_weight_cutoff: 0.0,
                dispersion: "none".into(),
                ..Default::default()
            };
            let numbers = vec![1; positions.len()];
            let setup = prepare_molecular(&numbers, &positions, &method).unwrap();
            let reference = run_molecular_reference(&setup).unwrap();
            assert!(reference.scf.converged);
            let o = [reference.scf.n_alpha, reference.scf.n_beta];
            let n = setup.basis.nao_sph;
            let weights = Pt2Coefficients {
                same_spin: 0.31,
                opposite_spin: 0.47,
            };
            let config = ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            };
            let analytic = correlated_molecular_axial_response(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                &setup.options,
                weights,
                Vec3::zero(),
                false,
                2048,
                &config,
            )
            .unwrap();
            assert!(analytic.response.converged);
            println!("{xc} U={unrestricted} AAT norm={:.12e} max amplitude={:.12e} connection={:.12e} cancellation={:.12e}",
            analytic.norm,analytic.amplitude_residuals.iter().copied().fold(0.0_f64,f64::max),
            analytic.connection_defects.iter().copied().fold(0.0_f64,f64::max),
            analytic.cancellation_defects.iter().copied().fold(0.0_f64,f64::max));
            let base = Wavefunction {
                positions: positions.clone(),
                orbitals: reference.scf.mo_coeff.each_ref().map(complex),
                amplitudes: std::array::from_fn(|_| Vec::new()),
                phase: Complex::ONE,
            };
            let real_integrals = eri_incore(&setup.basis, CoulombKernel::Coulomb, 2048).unwrap();
            let g: Vec<_> = real_integrals
                .as_slice()
                .iter()
                .copied()
                .map(Complex::from)
                .collect();
            let magnetic = |b: usize, h: f64| {
                let mut field = [0.0; 3];
                field[b] = h;
                let one = one_electron_series(
                    &setup.molecule,
                    &setup.basis,
                    &FieldPath {
                        electric: [0.0; 3],
                        magnetic: field,
                        origin: [0.0; 3],
                        london: false,
                        order: 2,
                    },
                )
                .unwrap();
                let core = std::array::from_fn(|_| {
                    let mut m = ComplexMatrix::zeros(n, n);
                    for k in 1..=2 {
                        for i in 0..n {
                            for j in 0..n {
                                m.re[(i, j)] += one.core[k].re[(i, j)];
                                m.im[(i, j)] += one.core[k].im[(i, j)];
                            }
                        }
                    }
                    m
                });
                // At zero vector potential this SCF uses the same current-
                // invariant XC model; physical B enters the explicit common-
                // origin one-electron operators instead of changing the AO span.
                let value = magnetic_field_scf_with_core(
                    &setup.molecule,
                    &setup.basis,
                    &setup.functional,
                    setup.grid.as_ref(),
                    &reference.scf,
                    &setup.options,
                    &MagneticFieldOptions {
                        field: [0.0; 3],
                        spin_zeeman: false,
                        series_order: Some(2),
                        series_tolerance: 1e-15,
                    },
                    &core,
                )
                .unwrap();
                assert!(value.converged);
                assert!(complex_pt2::finite_channels(
                    n,
                    &g,
                    &value.coefficients,
                    &value.orbital_energies,
                    o
                )
                .iter()
                .all(|x| x.is_finite()));
                let mut state = Wavefunction {
                    positions: positions.clone(),
                    amplitudes: complex_pt2::finite_amplitudes(
                        n,
                        &g,
                        &value.coefficients,
                        &value.orbital_energies,
                        o,
                    ),
                    orbitals: value.coefficients,
                    phase: Complex::ONE,
                };
                align(&setup, &base, &mut state, o);
                state
            };
            let nuclear = |x: usize, h: f64| {
                let mut shifted = positions.clone();
                shifted[x / 3][x % 3] += h;
                let system = prepare_molecular(&numbers, &shifted, &method).unwrap();
                let value = run_molecular_reference(&system).unwrap().scf;
                assert!(value.converged);
                let orbitals = value.mo_coeff.each_ref().map(complex);
                let eri = eri_incore(&system.basis, CoulombKernel::Coulomb, 2048).unwrap();
                let integrals: Vec<_> = eri.as_slice().iter().copied().map(Complex::from).collect();
                let mut state = Wavefunction {
                    positions: shifted,
                    amplitudes: complex_pt2::finite_amplitudes(
                        n,
                        &integrals,
                        &orbitals,
                        &value.mo_energy,
                        o,
                    ),
                    orbitals,
                    phase: Complex::ONE,
                };
                align(&setup, &base, &mut state, o);
                state
            };
            let mut cache = HashMap::new();
            let mut failures = Vec::new();
            for x in 0..3 * positions.len() {
                let names: Vec<_> = (0..3)
                    .flat_map(|b| {
                        (0..5).map(move |c| {
                            format!("{xc} U={unrestricted} AAT x={x} B={b} channel={c}")
                        })
                    })
                    .collect();
                components::fit_components(
                    &names,
                    &[
                        [0.02, 0.015, 0.01, 0.0075],
                        [0.005, 0.00375, 0.0025, 0.001875],
                        [0.04, 0.03, 0.02, 0.015],
                    ],
                    |steps| {
                        let mut errors = vec![[0.0; 4]; 15];
                        for (k, &h) in steps.iter().enumerate() {
                            let r = [nuclear(x, -h), nuclear(x, h)];
                            for b in 0..3 {
                                let bh = 0.03 * h;
                                let fields = cache
                                    .entry((b, h.to_bits()))
                                    .or_insert_with(|| [magnetic(b, -bh), magnetic(b, bh)]);
                                let targets = [
                                    analytic.response.value[x][b],
                                    analytic.parent[x][b],
                                    analytic.channel_bilinears[0][x][b],
                                    analytic.channel_bilinears[1][x][b],
                                    analytic.channel_bilinears[2][x][b],
                                ];
                                let mut mixed = [0.0; 5];
                                for ir in 0..2 {
                                    for ib in 0..2 {
                                        let cross = cross(&setup, &r[ir], &fields[ib]);
                                        let sign = if ir == ib { 1.0 } else { -1.0 };
                                        for c in 0..5 {
                                            mixed[c] += sign
                                                * slater::slater_overlap(
                                                    &determinants(&r[ir], n, o, c, weights),
                                                    &determinants(&fields[ib], n, o, c, weights),
                                                    &cross,
                                                )
                                                .im;
                                        }
                                    }
                                }
                                for c in 0..5 {
                                    errors[5 * b + c][k] =
                                        (mixed[c] / (4.0 * h * bh) - targets[c]).abs();
                                }
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
            }
            components::assert_no_failures(
                "all molecular normalized MP1 axial tensor entries",
                &failures,
            );
        }
    }
}
