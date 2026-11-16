// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every Cartesian electric derivative of a genuinely complex molecular
//! spinor state. A small fixed, nonsymmetric spatial quadrature isolates the
//! analytic density/current Hessian; spatial chemical convergence is separate.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::grid::{becke_weight_gradient, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::periodic::spinor_scf::{pauli_components, run_spinor_scf, SpinorScfConfig};
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;
use dft_rs::response::nuclear::spinor_electric_response;
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::{resolve, FunctionalSpec};

fn evaluate(xc: &str, field: [f64; 3], response: bool) -> (Vec<f64>, Vec<f64>) {
    let numbers = vec![1; 4];
    let positions = vec![
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
    ];
    let method = MolecularMethod {
        xc: Some("hf".into()),
        basis: "sto-3g".into(),
        field: Some(Vec3::new(field[0], field[1], field[2])),
        conv_tol: 1e-13,
        conv_tol_grad: Some(1e-12),
        conv_tol_dp: 1e-12,
        max_iter: 300,
        grid_weight_cutoff: 0.0,
        ..Default::default()
    };
    let setup = prepare_molecular(&numbers, &positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(xc.into())).unwrap();
    let points = vec![
        Vec3::new(0.44, 0.21, 0.72),
        Vec3::new(1.36, 0.49, 0.36),
        Vec3::new(0.08, 1.55, 0.86),
        Vec3::new(1.21, -0.13, 0.33),
        Vec3::new(0.61, 0.66, 2.0),
    ];
    let raw_weights = vec![0.043, 0.067, 0.039, 0.051, 0.062];
    let owner = vec![0, 1, 2, 1, 3];
    let centres: Vec<_> = setup.molecule.atoms.iter().map(|a| a.position).collect();
    let aij = MolecularGrid::becke_aij(&setup.molecule);
    let weights = points
        .iter()
        .enumerate()
        .map(|(i, &p)| raw_weights[i] * becke_weight_gradient(&centres, &aij, 4, owner[i], p).0)
        .collect();
    let grid = MolecularGrid {
        points,
        weights,
        raw_weights,
        atom_of: owner,
    };
    let mut operator = BlochMatrix::zeros(8, 8);
    for i in 0..4 {
        operator.re[(i, i)] = -0.013 * (i + 1) as f64;
        operator.re[(4 + i, 4 + i)] = -operator.re[(i, i)];
        operator.re[(i, 4 + i)] = 0.007 * (i + 1) as f64;
        operator.re[(4 + i, i)] = operator.re[(i, 4 + i)];
        operator.im[(i, 4 + i)] = 0.009 * (4 - i) as f64;
        operator.im[(4 + i, i)] = -operator.im[(i, 4 + i)];
    }
    let backend = MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &functional,
        Some(&grid),
        &seed.scf,
        &setup.options,
        operator.clone(),
        [0.2, 0.3, 0.7],
        MolecularSpinorFunctionalModel::SpinCovariant,
        1e-11,
    )
    .unwrap();
    let scf = run_spinor_scf(
        &backend,
        &SpinorScfConfig {
            energy_tolerance: 1e-13,
            density_tolerance: 1e-12,
            orbital_tolerance: 1e-12,
            max_iter: 300,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(scf.converged);
    let pauli = pauli_components(&scf.density[0]).unwrap();
    let dipole = dft_rs::integrals::dipole_matrices(&setup.basis, Vec3::zero())
        .map(|d| -pauli[0].re.frobenius_dot(&d))
        .to_vec();
    if !response {
        return (dipole, Vec::new());
    }
    let frequencies = [
        Complex::ZERO,
        Complex::new(0.11, 0.02),
        Complex::new(-0.11, 0.02),
        Complex::new(0.0, 0.13),
    ];
    let values = spinor_electric_response(
        &setup.molecule,
        &setup.basis,
        Some(&grid),
        &functional,
        &scf,
        &setup.options,
        &operator,
        &frequencies,
        &ResponseSolverConfig {
            tolerance: 1e-11,
            ..Default::default()
        },
    )
    .unwrap();
    let values = values.values;
    for value in &values {
        assert!(value.residuals.iter().all(|r| *r < 1e-10));
        println!(
            "{xc} spinor frequency {:?} residuals {:?}",
            value.frequency, value.residuals
        );
    }
    for a in 0..3 {
        for b in 0..3 {
            // Causal reality holds at opposite signed frequencies even when
            // time reversal is broken; no same-frequency transpose is imposed.
            assert!((values[1].alpha[a][b] - values[2].alpha[a][b].conj()).abs() < 1e-9);
            assert!(values[0].alpha[a][b].im.abs() < 1e-9);
        }
    }
    (
        dipole,
        values[0].alpha.iter().flatten().map(|v| v.re).collect(),
    )
}

#[test]
fn covariant_complex_susceptibility_fits_every_dipole_field_derivative() {
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0"] {
        if std::env::var("DFT_RS_TEST_MODEL")
            .ok()
            .is_some_and(|v| v != xc)
        {
            continue;
        }
        let (_, alpha) = evaluate(xc, [0.0; 3], true);
        let stages = [
            [0.01, 0.0075, 0.005, 0.00375],
            [0.002, 0.0015, 0.001, 0.00075],
            [0.02, 0.015, 0.01, 0.0075],
        ];
        let mut failures = Vec::new();
        for b in 0..3 {
            let names = (0..3)
                .map(|a| format!("{xc} complex alpha({a},{b})"))
                .collect::<Vec<_>>();
            components::fit_components(
                &names,
                &stages,
                |steps| {
                    let mut errors = vec![[0.0; 4]; 3];
                    for (k, &h) in steps.iter().enumerate() {
                        let sample = |x| {
                            let mut f = [0.0; 3];
                            f[b] = x;
                            evaluate(xc, f, false).0
                        };
                        let plus = sample(h);
                        let minus = sample(-h);
                        for a in 0..3 {
                            errors[a][k] =
                                ((plus[a] - minus[a]) / (2.0 * h) - alpha[3 * a + b]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
        components::assert_no_failures(&format!("{xc} full complex alpha"), &failures);
    }
}
