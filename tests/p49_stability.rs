// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::{eri_incore, CoulombKernel, EriTensor};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::response::complex::Complex;
use dft_rs::response::{
    molecular_stability, rotated_density, stability_from_state, stabilize, ResponseSolverConfig,
    StabilityChannel, StabilityOptions, StabilizeOptions,
};
use dft_rs::scf::{run_scf, ScfOptions, ScfResult};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};
#[path = "support/convergence.rs"]
mod convergence;

fn setup(
    helium: bool,
    distance: f64,
    functional: &str,
) -> (
    Molecule,
    MolBasis,
    ScfOptions,
    ResolvedFunctional,
    Option<MolecularGrid>,
    ScfResult,
) {
    let mut mol = Molecule::new(vec![
        Atom {
            z: if helium { 2 } else { 1 },
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, distance),
        },
    ]);
    if helium {
        mol.charge = 1;
    }
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let f = resolve(&FunctionalSpec::Combined(functional.into())).unwrap();
    let opts = ScfOptions {
        max_iter: 250,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        ..Default::default()
    };
    let grid = if f.parts.is_empty() {
        None
    } else {
        Some(
            MolecularGrid::build(
                &mol,
                &GridOptions {
                    level: 1,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    };
    let scf = run_scf(&mol, &mb, &opts, &f, grid.as_ref()).unwrap();
    assert!(scf.converged);
    (mol, mb, opts, f, grid, scf)
}

// Independent determinant energy oracle: explicit complex occupied orbitals
// and four-index ERI sums. No response Hessian, Fock build or TDDFT product.
fn determinant_energy(
    theta: f64,
    imaginary: bool,
    triplet: bool,
    mb: &MolBasis,
    f: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    eri: &EriTensor,
    lr: Option<&EriTensor>,
) -> f64 {
    let n = mb.nao_sph;
    let mut p = vec![vec![Complex::ZERO; n * n]; 2];
    let mut re = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
    for s in 0..2 {
        let sign = if triplet && s == 1 { -1.0 } else { 1.0 };
        let c: Vec<_> = (0..n)
            .map(|mu| {
                let o = scf.mo_coeff[s][(mu, 0)] * theta.cos();
                let v = sign * scf.mo_coeff[s][(mu, 1)] * theta.sin();
                if imaginary {
                    Complex::new(o, v)
                } else {
                    Complex::new(o + v, 0.0)
                }
            })
            .collect();
        for mu in 0..n {
            for nu in 0..n {
                p[s][mu * n + nu] = c[mu] * c[nu].conj();
                re[s][(mu, nu)] = p[s][mu * n + nu].re;
            }
        }
    }
    let mut energy = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            energy += (p[0][mu * n + nu] + p[1][mu * n + nu]).re * scf.h_core[(nu, mu)];
            for la in 0..n {
                for si in 0..n {
                    let total = p[0][mu * n + nu] + p[1][mu * n + nu];
                    let other = p[0][la * n + si] + p[1][la * n + si];
                    energy += 0.5 * (total * other).re * eri.get(mu, nu, la, si);
                    for spin in &p {
                        let product = (spin[mu * n + nu] * spin[la * n + si]).re;
                        energy -= 0.5 * f.hybrid.cx * product * eri.get(nu, la, mu, si);
                        if let Some(lr) = lr {
                            energy -= 0.5 * f.hybrid.clr * product * lr.get(nu, la, mu, si);
                        }
                    }
                }
            }
        }
    }
    if let Some(g) = grid {
        energy += dft_rs::ks::xc_contribution(f, mb, g, [&re[0], &re[1]])
            .unwrap()
            .exc;
    }
    energy
}

#[test]
fn all_real_and_imaginary_spin_curvatures_have_four_step_energy_oracles() {
    let steps = [0.01, 0.005, 0.0025, 0.00125];
    for name in ["hf", "svwn5", "pbe", "b3lyp", "cam-b3lyp", "r2scan"] {
        let (mol, mb, opts, f, grid, scf) = setup(true, 1.75, name);
        let report = stability_from_state(
            &mol,
            &mb,
            &f,
            grid.as_ref(),
            &scf,
            &opts,
            &StabilityOptions {
                n_roots: 1,
                tolerance: 1e-10,
                max_iterations: 500,
                channels: vec![
                    StabilityChannel::InternalReal,
                    StabilityChannel::ExternalSpin,
                    StabilityChannel::ExternalComplex,
                    StabilityChannel::ExternalComplexSpin,
                ],
                ..Default::default()
            },
        )
        .unwrap();
        assert!(report.converged);
        let eri = eri_incore(&mb, CoulombKernel::Coulomb, 32).unwrap();
        let lr = if f.hybrid.clr == 0.0 {
            None
        } else {
            Some(
                eri_incore(
                    &mb,
                    CoulombKernel::Erf {
                        omega: f.hybrid.omega,
                    },
                    32,
                )
                .unwrap(),
            )
        };
        for block in &report.value.blocks {
            assert_eq!(block.dimension, 1);
            let imaginary = matches!(
                block.channel,
                StabilityChannel::ExternalComplex | StabilityChannel::ExternalComplexSpin
            );
            let triplet = matches!(
                block.channel,
                StabilityChannel::ExternalSpin | StabilityChannel::ExternalComplexSpin
            );
            let exact = block.modes[0].energy_curvature;
            let energy = |t| {
                determinant_energy(
                    t,
                    imaginary,
                    triplet,
                    &mb,
                    &f,
                    grid.as_ref(),
                    &scf,
                    &eri,
                    lr.as_ref(),
                )
            };
            let center = energy(0.0);
            let values: Vec<_> = steps
                .iter()
                .map(|h| (energy(*h) - 2.0 * center + energy(-*h)) / (h * h))
                .collect();
            let errors: Vec<_> = values.iter().map(|v| (v - exact).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            println!(
                "stability {name} {} h={steps:?} errors={errors:?} p={:.12} R2={:.12} \
                C={:.12e} log_rmse={:.12e}",
                block.name, fit.exponent, fit.r_squared, fit.prefactor, fit.log_rmse
            );
            assert!((fit.exponent - 2.0).abs() < 0.02);
            assert!(fit.r_squared > 0.99999);
            for pair in errors.windows(2) {
                assert!((pair[0] / pair[1] / 4.0 - 1.0).abs() < 0.03);
            }
            let richardson = (4.0 * values[3] - values[2]) / 3.0;
            assert!(
                (richardson - exact).abs() < 1e-6,
                "{name} {} {richardson} {exact}",
                block.name
            );
        }
    }
}

#[test]
fn stretched_restricted_hydrogen_is_spin_unstable_but_response_converges() {
    for (r, expected) in [(1.4, "stable"), (4.5, "unstable")] {
        let (mol, mb, opts, f, grid, scf) = setup(false, r, "hf");
        let report = molecular_stability(
            &mol,
            &mb,
            &f,
            grid.as_ref(),
            &scf,
            &opts,
            2,
            &ResponseSolverConfig::default(),
        )
        .unwrap();
        assert!(report.converged);
        assert_eq!(report.value.classification, expected);
        let triplet = report
            .value
            .blocks
            .iter()
            .find(|b| b.name == "external-spin")
            .unwrap();
        assert_eq!(triplet.classification, expected);
        assert!(report.residuals.iter().all(|r| *r < 1e-10));
    }
}

#[test]
fn unrestricted_one_electron_curvatures_and_invalid_inputs() {
    let (mut mol, mb, opts, f, grid, _) = setup(true, 1.75, "hf");
    mol.charge = 2;
    mol.multiplicity = 2;
    let mut scf = run_scf(&mol, &mb, &opts, &f, None).unwrap();
    let report = molecular_stability(
        &mol,
        &mb,
        &f,
        grid.as_ref(),
        &scf,
        &opts,
        1,
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    assert_eq!(report.value.blocks.len(), 1);
    let gap = scf.mo_energy[0][1] - scf.mo_energy[0][0];
    // The occupied one-electron self-interaction cancels including its orbital response.
    let (bare, _) = dft_rs::linalg::symmetric_eigen(
        &scf.mo_coeff[0]
            .transpose()
            .matmul_seq(&scf.h_core)
            .matmul_seq(&scf.mo_coeff[0]),
    )
    .unwrap();
    assert!(gap.is_finite());
    for b in report.value.blocks {
        assert!((b.modes[0].energy_curvature - 2.0 * (bare[1] - bare[0])).abs() < 1e-9);
    }
    assert!(molecular_stability(
        &mol,
        &mb,
        &f,
        None,
        &scf,
        &opts,
        0,
        &ResponseSolverConfig::default()
    )
    .is_err());
    scf.converged = false;
    assert!(molecular_stability(
        &mol,
        &mb,
        &f,
        None,
        &scf,
        &opts,
        1,
        &ResponseSolverConfig::default()
    )
    .is_err());
}

#[test]
fn following_lowers_internal_uhf_saddle_preserves_class_and_rechecks_stability() {
    let (mol, mb, mut opts, f, grid, rhf) = setup(false, 4.5, "hf");
    opts.reference = dft_rs::scf::Reference::Uhf;
    let mut saddle = rhf.clone();
    saddle.unrestricted = true;
    let options = StabilizeOptions {
        analysis: StabilityOptions {
            tolerance: 1e-10,
            max_iterations: 500,
            ..Default::default()
        },
        ..Default::default()
    };
    let report = stabilize(&mol, &mb, &f, grid.as_ref(), &saddle, &opts, &options).unwrap();
    assert!(report.converged);
    assert!(report.stabilized);
    assert!(report.final_state.unrestricted);
    assert_eq!(report.descents.len(), 1);
    for d in &report.descents {
        assert_eq!(d.channel, StabilityChannel::UnrestrictedInternal);
        assert!(d.energy_before - d.energy_after >= 1e-8);
    }
    assert!(saddle.energy - report.final_state.energy > 0.1);
    let stationary = &report.initial.value.blocks[0];
    let seed = rotated_density(
        &saddle,
        stationary.channel,
        &stationary.modes[0].amplitudes,
        0.3,
    )
    .unwrap();
    for p in &seed {
        assert!(p.matmul_seq(&saddle.overlap).matmul_seq(p).max_abs_diff(p) < 1e-12);
        assert!((p.frobenius_dot(&saddle.overlap) - 1.0).abs() < 1e-12);
    }
    // Independent broken-symmetry SCF seed converges to the same lower basin.
    let independent_opts = ScfOptions {
        guess_mix: 23.0,
        ..opts.clone()
    };
    let independent = run_scf(&mol, &mb, &independent_opts, &f, None).unwrap();
    assert!((independent.energy - report.final_state.energy).abs() < 1e-11);
    let restricted =
        stabilize(&mol, &mb, &f, None, &rhf, &ScfOptions::default(), &options).unwrap();
    assert!(restricted.descents.is_empty());
    assert!(!restricted.stabilized);
    assert!(!restricted.final_state.unrestricted);
    saddle.converged = false;
    assert!(stabilize(&mol, &mb, &f, None, &saddle, &opts, &options).is_err());
}

#[test]
fn resolution_channel_contract_and_unconverged_numbers_are_explicit() {
    let (mol, mb, opts, f, grid, scf) = setup(true, 1.75, "hf");
    let options = StabilityOptions {
        resolution: Some(1000.0),
        ..Default::default()
    };
    let report = stability_from_state(&mol, &mb, &f, grid.as_ref(), &scf, &opts, &options).unwrap();
    assert_eq!(report.value.classification, "indefinite");
    assert!(!report.value.stable);
    assert!(report.converged);
    for b in &report.value.blocks {
        assert!(!b.dominant.is_empty());
        assert_eq!(b.dominant[0].from, 0);
        assert_eq!(b.dominant[0].to, 1);
        assert_eq!(b.eigenvector.len(), 1);
    }
    assert!(StabilityOptions {
        resolution: Some(1e-12),
        ..Default::default()
    }
    .validate()
    .is_err());
    assert!(stability_from_state(
        &mol,
        &mb,
        &f,
        None,
        &scf,
        &opts,
        &StabilityOptions {
            channels: vec![StabilityChannel::UnrestrictedInternal],
            ..Default::default()
        }
    )
    .is_err());
}
