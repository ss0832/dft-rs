// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving-basis fixed-spectrum energy derivatives against independent solvers.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::gradient::assemble_gradient;
use dft_rs::hessian::analytic::analytic_hessian_full;
use dft_rs::math::Vec3;
use dft_rs::response::{nuclear::nuclear_taylor, ResponseSolverConfig};
use dft_rs::scf::occupations::FixedOccupations;

#[test]
fn four_moving_nuclear_orders_fit_independent_energy_gradient_and_hessian() {
    for unrestricted in [false, true] {
        let method = MolecularMethod {
            xc: Some("hf".into()),
            basis: "6-31g".into(),
            unrestricted,
            dispersion: "none".into(),
            max_iter: 500,
            conv_tol: 1e-14,
            conv_tol_grad: Some(1e-12),
            conv_tol_dp: 1e-12,
            field: Some(Vec3::new(0.001, -0.002, 0.003)),
            fixed_occupations: Some(FixedOccupations {
                populations: [
                    vec![0.68, 0.22, 0.08, 0.02],
                    if unrestricted {
                        vec![0.61, 0.26, 0.10, 0.03]
                    } else {
                        vec![0.68, 0.22, 0.08, 0.02]
                    },
                ],
            }),
            ..MolecularMethod::default()
        };
        let at = |t: f64| {
            let s = prepare_molecular(
                &[1, 1],
                &[[0.0, 0.0, -0.8], [0.2, 0.1, 0.8 + 0.1 * t]],
                &method,
            )
            .unwrap();
            let r = run_molecular_reference(&s).unwrap();
            assert!(r.scf.converged);
            let gradient = assemble_gradient(
                &s.molecule,
                &s.basis,
                &s.functional,
                s.grid.as_ref(),
                &r.scf,
                &s.options,
            )
            .unwrap();
            let hessian = analytic_hessian_full(
                &s.molecule,
                &s.basis,
                &s.options,
                &s.functional,
                &r.scf,
                s.grid.as_ref(),
            )
            .unwrap();
            (
                r.scf.energy,
                0.1 * gradient[1].z,
                0.01 * hessian.hessian[(5, 5)],
            )
        };
        let setup =
            prepare_molecular(&[1, 1], &[[0.0, 0.0, -0.8], [0.2, 0.1, 0.8]], &method).unwrap();
        let reference = run_molecular_reference(&setup).unwrap();
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..ResponseSolverConfig::default()
        };
        let report = nuclear_taylor(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            &[[0.0; 3], [0.0, 0.0, 0.1]],
            4,
            &config,
        )
        .unwrap();
        println!("fixed nuclear unrestricted={unrestricted} converged={} population={:?} metric={:?} commutator={:?} number={:?} equation={:?}",report.converged,report.value.fixed_population.as_ref().unwrap().population_defects,report.value.metric_defects,report.value.stationarity_defects,report.value.particle_number_coefficients,report.residuals);
        assert!(report.converged);
        let base = at(0.0);
        let factorial = [1.0, 1.0, 2.0, 6.0, 24.0];
        assert!((base.1 - report.value.energy_coefficients[1]).abs() < 1e-11);
        assert!((base.2 - 2.0 * report.value.energy_coefficients[2]).abs() < 1e-11);
        let steps = [0.4, 0.2, 0.1, 0.05];
        let mut estimates: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for h in steps {
            let plus = at(h);
            let minus = at(-h);
            estimates[0].push((plus.0 - minus.0) / (2.0 * h));
            estimates[1].push((plus.1 - minus.1) / (2.0 * h));
            estimates[2].push((plus.2 - minus.2) / (2.0 * h));
            estimates[3].push((plus.2 - 2.0 * base.2 + minus.2) / (h * h));
        }
        for n in 1..=4 {
            let analytic = factorial[n] * report.value.energy_coefficients[n];
            let errors: Vec<_> = estimates[n - 1]
                .iter()
                .map(|v| (v - analytic).abs())
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson =
                ((4.0 * estimates[n - 1][3] - estimates[n - 1][2]) / 3.0 - analytic).abs();
            println!("fixed nuclear unrestricted={unrestricted} order={n} analytic={analytic:.14e} steps={steps:?} errors={errors:?} p={:.12} R2={:.12} C={:.12e} log_rmse={:.12e} Richardson={richardson:.12e}",fit.exponent,fit.r_squared,fit.prefactor,fit.log_rmse);
            assert!((fit.exponent - 2.0).abs() < 0.025);
            assert!(fit.r_squared > 0.9999);
            assert!(richardson < 2e-7 + 1e-8 * analytic.abs());
        }
    }
}
