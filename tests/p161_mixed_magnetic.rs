// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Mixed second and fourth magnetic derivatives from independently converged
//! finite-field SCF energies. Every sign is computed, without symmetry copying.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{data, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::response::magnetic_mixed::mixed_magnetic_energy;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::magnetic::{magnetic_field_scf, MagneticFieldOptions};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

#[test]
fn mixed_second_and_fourth_magnetic_energy_have_four_step_error_fits() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ]);
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let options = ScfOptions {
        e_tol: 1e-14,
        p_tol: 1e-14,
        max_iter: 250,
        ..Default::default()
    };
    let reference = run_scf(&molecule, &basis, &options, &functional, None).unwrap();
    let u = [0.7, 0.1, 0.0];
    let v = [0.1, 0.4, 0.2];
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let second = mixed_magnetic_energy(
        &molecule,
        &basis,
        &functional,
        None,
        &reference,
        &options,
        Vec3::zero(),
        true,
        &[u, v],
        &config,
    )
    .unwrap();
    let fourth = mixed_magnetic_energy(
        &molecule,
        &basis,
        &functional,
        None,
        &reference,
        &options,
        Vec3::zero(),
        true,
        &[u, u, v, v],
        &config,
    )
    .unwrap();
    assert!(second.derivative.abs() > 1e-6 && fourth.derivative.abs() > 1e-7);
    println!("mixed magnetic exact second={:.15e} fourth={:.15e} residuals=[{:.12e},{:.12e}] projector=[{:.12e},{:.12e}] stationarity=[{:.12e},{:.12e}] roundoff=[{:.12e},{:.12e}]",second.derivative,fourth.derivative,
        second.response_residual,fourth.response_residual,second.projector_defect,fourth.projector_defect,
        second.stationarity_defect,fourth.stationarity_defect,second.accumulation_roundoff,fourth.accumulation_roundoff);
    let steps = [0.32, 0.24, 0.16, 0.08];
    let mut estimates = [Vec::new(), Vec::new()];
    for h in steps {
        let mut energies = [[0.0; 3]; 3];
        for (i, a) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
            for (j, b) in [-1.0, 0.0, 1.0].into_iter().enumerate() {
                let field = std::array::from_fn(|k| h * (a * u[k] + b * v[k]));
                let state = magnetic_field_scf(
                    &molecule,
                    &basis,
                    &functional,
                    None,
                    &reference,
                    &options,
                    &MagneticFieldOptions {
                        field,
                        spin_zeeman: false,
                        series_order: Some(16),
                        series_tolerance: 1e-14,
                    },
                )
                .unwrap();
                assert!(state.converged);
                energies[i][j] = state.energy;
            }
        }
        estimates[0].push(
            (energies[2][2] - energies[2][0] - energies[0][2] + energies[0][0]) / (4.0 * h * h),
        );
        let weights = [1.0, -2.0, 1.0];
        estimates[1].push(
            (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| weights[i] * weights[j] * energies[i][j])
                .sum::<f64>()
                / h.powi(4),
        );
    }
    for (index, (order, target)) in [(2, second.derivative), (4, fourth.derivative)]
        .into_iter()
        .enumerate()
    {
        let errors: Vec<_> = estimates[index]
            .iter()
            .map(|x| (x - target).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[index][3] - estimates[index][2]) / 3.0 - target).abs();
        println!("mixed magnetic order={order} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
        assert!(rich < 1e-6);
    }
}
