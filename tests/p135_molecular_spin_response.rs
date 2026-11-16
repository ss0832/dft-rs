// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Number-constrained molecular spin response versus independent nonlinear
//! spin-transfer references, with four distinct steps for every parent.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::linalg::Matrix;
use dft_rs::response::ensemble::EnsembleSolver;
use dft_rs::response::ResponseSolverConfig;

fn trace(a: &Matrix, b: &Matrix) -> f64 {
    (0..a.rows)
        .flat_map(|i| (0..a.cols).map(move |j| a[(i, j)] * b[(j, i)]))
        .sum()
}

#[test]
fn generic_spin_stiffness_and_field_density_match_nonlinear_occupation_paths() {
    for xc in ["svwn5", "pbe", "tpss", "pbe0"] {
        let method = MolecularMethod {
            xc: Some(xc.into()),
            unrestricted: true,
            electronic_temperature: 150000.,
            grid_level: 4,
            conv_tol: 1e-13,
            conv_tol_grad: Some(1e-11),
            conv_tol_dp: 1e-12,
            dispersion: "none".into(),
            ..MolecularMethod::default()
        };
        let at = |delta: f64| {
            let mut options = method.clone();
            options.electron_offset = [delta, -delta];
            let setup =
                prepare_molecular(&[1, 1], &[[0., 0., -0.7], [0., 0., 0.7]], &options).unwrap();
            let reference = run_molecular_reference(&setup).unwrap();
            (setup, reference)
        };
        let (setup, reference) = at(0.);
        let solver = EnsembleSolver::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
        )
        .unwrap();
        let response = solver
            .spin_susceptibility(2., &ResponseSolverConfig::default())
            .unwrap();
        let steps = [0.08, 0.04, 0.02, 0.01];
        let mut estimates = Vec::new();
        for h in steps {
            let (_, plus) = at(h);
            let (_, minus) = at(-h);
            let mu = |r: &dft_rs::scf::ScfResult| {
                r.chemical_potentials[0].unwrap() - r.chemical_potentials[1].unwrap()
            };
            estimates.push((mu(&plus.scf) - mu(&minus.scf)) / (2. * h));
        }
        let errors: Vec<_> = estimates
            .iter()
            .map(|x| (x - response.stiffness).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4. * estimates[3] - estimates[2]) / 3. - response.stiffness).abs();
        let populations = response
            .delta_density
            .each_ref()
            .map(|p| trace(p, &reference.scf.overlap));
        let moment = -0.5 * (populations[0] - populations[1]);
        println!("{xc} chi={:.12e} bare={:.12e} stiffness={:.12e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e} residual={:.12e} number={:.12e} moment={:.12e}",response.susceptibility,response.bare_susceptibility,response.stiffness,fit.exponent,fit.r_squared,response.response_residual,(populations[0]+populations[1]).abs(),(moment-response.susceptibility).abs());
        assert!((fit.exponent - 2.).abs() < 0.05 && fit.r_squared > 0.9999);
        assert!(richardson < 1e-7);
        assert!((populations[0] + populations[1]).abs() < 1e-12);
        assert!((moment - response.susceptibility).abs() < 1e-12);
        assert!(response.response_residual < 1e-10);
        let scaled = solver
            .spin_susceptibility(3., &ResponseSolverConfig::default())
            .unwrap();
        assert!((scaled.susceptibility - 2.25 * response.susceptibility).abs() < 1e-12);
        let zero = solver
            .spin_susceptibility(0., &ResponseSolverConfig::default())
            .unwrap();
        assert_eq!(zero.susceptibility, 0.);
        assert!(zero
            .delta_density
            .iter()
            .flat_map(|p| p.as_slice())
            .all(|x| *x == 0.));
    }
}
