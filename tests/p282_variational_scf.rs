// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Retained fixed-quantum-potential SCF failures at unchanged physical gates.
//! This is SCF acceptance; coupled derivative fits remain in p223.

use dft_rs::driver::{run, Controls, System, Value};

#[test]
fn fixed_quantum_potentials_converge_across_real_spin_constraints_and_grids() {
    for unrestricted in [false, true] {
        for (grid, lmax, displacement) in [(2usize, 0usize, 0.0), (2, 1, 0.04), (4, 1, 0.0)] {
            let system = System::molecule(
                vec![1, 1],
                vec![[0.0, 0.0, -0.7 + displacement], [0.0, 0.0, 0.7]],
            );
            let request = Value::object()
                .with("xc", "tpss")
                .with("unrestricted", unrestricted)
                .with("dispersion", "none")
                .with("basis", "sto-3g")
                .with("grid_level", grid)
                .with("grid_weight_cutoff", 0.0)
                .with("quantum_atoms", vec![1usize])
                .with("quantum_exponents", vec![4.0, 8.0])
                .with("quantum_lmax", lmax)
                .with("interspecies_correlation", "none")
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-12)
                .with("conv_tol_dp", 1e-12)
                .with("max_iter", 500usize);
            let report = run(
                "multicomponent",
                &system,
                &Controls::from_value(&request).unwrap(),
            )
            .unwrap();
            assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
            for key in ["electronic_orbital_residual", "electronic_density_residual"] {
                let residual = report.get(key).and_then(Value::as_f64).unwrap();
                println!("TPSS real unrestricted={unrestricted} grid={grid} lmax={lmax} displacement={displacement} {key}={residual:.12e}");
                assert!(residual < 1e-10);
            }
            let energy = report.get("energy").and_then(Value::as_f64).unwrap();
            let Value::Object(terms) = report.get("energy_components").unwrap() else {
                panic!("named energy terms are required");
            };
            let sum: f64 = terms.iter().map(|(_, v)| v.as_f64().unwrap()).sum();
            assert!((energy - sum).abs() < 1e-11);
        }
    }
}
