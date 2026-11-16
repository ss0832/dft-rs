// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed orbital populations: physical density, independent energies and forces.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::integrals::{eri_incore, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::scf::occupations::FixedOccupations;

fn method(xc: &str, unrestricted: bool) -> MolecularMethod {
    MolecularMethod {
        xc: Some(xc.into()),
        unrestricted,
        grid_level: 4,
        dispersion: "none".into(),
        conv_tol: 1e-13,
        conv_tol_grad: Some(1e-11),
        conv_tol_dp: 1e-12,
        max_iter: 500,
        fixed_occupations: Some(FixedOccupations {
            populations: [
                vec![0.8, 0.2],
                vec![
                    if unrestricted { 0.7 } else { 0.8 },
                    if unrestricted { 0.3 } else { 0.2 },
                ],
            ],
        }),
        ..Default::default()
    }
}

#[test]
fn populations_reproduce_closed_density_and_independent_energy_components() {
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0", "cam-b3lyp"] {
        for unrestricted in [false, true] {
            let m = method(xc, unrestricted);
            let setup = prepare_molecular(&[1, 1], &[[0., 0., -0.7], [0., 0., 0.7]], &m).unwrap();
            let state = run_molecular_reference(&setup).unwrap().scf;
            if xc == "hf" && unrestricted {
                let mut mixed = m.clone();
                mixed.guess_mix = Some(18.0);
                let mixed_setup =
                    prepare_molecular(&[1, 1], &[[0., 0., -0.7], [0., 0., 0.7]], &mixed).unwrap();
                let mixed_state = run_molecular_reference(&mixed_setup).unwrap().scf;
                assert!((mixed_state.energy - state.energy).abs() < 1e-11);
                for spin in 0..2 {
                    assert!(mixed_state.density[spin].max_abs_diff(&state.density[spin]) < 1e-10);
                }
            }
            assert!(state.fixed_occupation_residual.unwrap() < 1e-12);
            assert!(!state.has_integer_occupations());
            assert_eq!(state.homo_lumo(), (None, None));
            assert_eq!(state.electronic_entropy, 0.0);
            assert_eq!(state.chemical_potentials, [None; 2]);
            let overlap = state.overlap[(0, 1)];
            for spin in 0..2 {
                let f = &m.fixed_occupations.as_ref().unwrap().populations[spin];
                let mut exact = Matrix::zeros(2, 2);
                for i in 0..2 {
                    for j in 0..2 {
                        exact[(i, j)] = f[0] / (2. * (1. + overlap))
                            + f[1] * if i == j { 1. } else { -1. } / (2. * (1. - overlap));
                    }
                }
                assert!(state.density[spin].max_abs_diff(&exact) < 1e-10);
                assert!((state.density[spin].frobenius_dot(&state.overlap) - 1.).abs() < 1e-11);
                assert!(
                    state.mo_coeff[spin]
                        .transpose_matmul(&state.overlap.matmul_seq(&state.mo_coeff[spin]))
                        .max_abs_diff(&Matrix::identity(2))
                        < 1e-10
                );
                assert_eq!(state.occupations[spin], *f);
            }
            // Direct four-index AO contractions of the known closed-form
            // densities, independent of the SCF J/K engine and its energy.
            let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 16).unwrap();
            let lr = if setup.functional.hybrid.clr != 0. {
                Some(
                    eri_incore(
                        &setup.basis,
                        CoulombKernel::Erf {
                            omega: setup.functional.hybrid.omega,
                        },
                        16,
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            let p = state.p_total();
            let one = p.frobenius_dot(&state.h_core);
            let mut hartree = 0.;
            let mut exchange = 0.;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            hartree += 0.5 * p[(i, j)] * p[(k, l)] * eri.get(i, j, k, l);
                            for density in &state.density {
                                exchange -= 0.5
                                    * density[(i, k)]
                                    * density[(j, l)]
                                    * (setup.functional.hybrid.cx * eri.get(i, j, k, l)
                                        + setup.functional.hybrid.clr
                                            * lr.as_ref().map_or(0., |v| v.get(i, j, k, l)));
                            }
                        }
                    }
                }
            }
            let semilocal = if let Some(grid) = &setup.grid {
                dft_rs::ks::xc_contribution(
                    &setup.functional,
                    &setup.basis,
                    grid,
                    [&state.density[0], &state.density[1]],
                )
                .unwrap()
                .exc
            } else {
                0.
            };
            let oracle = one + hartree + exchange + semilocal + state.e_nuc;
            assert!((oracle - state.energy).abs() < 2e-11);
            assert!((semilocal - state.e_xc).abs() < 2e-11);
            // Explicit independent Bernoulli occupations of the four H2 spin
            // orbitals: average the exact determinant S^2 of all 16 states.
            let alpha = &state.occupations[0];
            let beta = &state.occupations[1];
            let mut spin_oracle = 0.;
            for bits in 0usize..16 {
                let weights = [alpha[0], alpha[1], beta[0], beta[1]];
                let occupied = std::array::from_fn::<_, 4, _>(|i| ((bits >> i) & 1) as f64);
                let probability: f64 = (0..4)
                    .map(|i| {
                        if occupied[i] == 1. {
                            weights[i]
                        } else {
                            1. - weights[i]
                        }
                    })
                    .product();
                let sz = 0.5 * (occupied[0] + occupied[1] - occupied[2] - occupied[3]);
                let exact = sz * (sz + 1.) + occupied[2] + occupied[3]
                    - occupied[0] * occupied[2]
                    - occupied[1] * occupied[3];
                spin_oracle += probability * exact;
            }
            assert!((state.s_squared - spin_oracle).abs() < 1e-10);
            println!("fixed energy {xc} unrestricted={unrestricted} energy={:.16e} one={one:.16e} Hartree={hartree:.16e} exchange={exchange:.16e} semilocal={semilocal:.16e} oracle_error={:.16e} map={:.16e} S2={:.16e}", state.energy, (oracle-state.energy).abs(), state.fixed_occupation_residual.unwrap(), state.s_squared);
        }
    }
}

#[test]
fn fixed_population_finite_field_analytic_forces_have_independent_four_step_fits() {
    let mut accepted = true;
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0"] {
        for unrestricted in [false, true] {
            let mut m = method(xc, unrestricted);
            m.field = Some(dft_rs::math::Vec3::new(0., 0., 0.0003));
            let at = |t| {
                let setup = prepare_molecular(
                    &[1, 1],
                    &[[0., 0., -0.7 - 0.5 * t], [0., 0., 0.7 + 0.5 * t]],
                    &m,
                )
                .unwrap();
                let state = run_molecular_reference(&setup).unwrap();
                (setup, state)
            };
            let (setup, state) = at(0.);
            let gradient = dft_rs::gradient::assemble_gradient(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &state.scf,
                &setup.options,
            )
            .unwrap();
            let analytic = 0.5 * (gradient[1].z - gradient[0].z);
            let steps = [0.04, 0.02, 0.01, 0.005];
            let estimates: Vec<_> = steps
                .iter()
                .map(|h| (at(*h).1.total_energy() - at(-*h).1.total_energy()) / (2. * h))
                .collect();
            let errors: Vec<_> = estimates.iter().map(|v| (v - analytic).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson = ((4. * estimates[3] - estimates[2]) / 3. - analytic).abs();
            println!("fixed force {xc} unrestricted={unrestricted} analytic={analytic:.16e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} prefactor={:.9e} log_rmse={:.9e} Richardson={richardson:.16e}",fit.exponent,fit.r_squared,fit.prefactor,fit.log_rmse);
            accepted &=
                (fit.exponent - 2.).abs() < 0.05 && fit.r_squared > 0.9999 && richardson < 2e-7;
        }
    }
    assert!(
        accepted,
        "every fixed-population force fit must pass the unchanged gates"
    );
}
