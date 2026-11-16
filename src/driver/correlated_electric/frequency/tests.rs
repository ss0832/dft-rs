// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::driver::double_hybrid::moment_tests::{correlated_controls, system, EXPONENT, ORIGIN};
use crate::driver::{run, Controls, System};
use crate::math::Vec3;

#[test]
fn assembled_pt2_frequency_recovers_all_static_dipole_channels() {
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-dynamic-pt2-static-limit-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    for unrestricted in [false, true] {
        let system = system(unrestricted);
        let controls = correlated_controls("custom", unrestricted, basis.to_str().unwrap());
        let input = Controls::from_value(&controls).unwrap();
        let prepared = crate::driver::double_hybrid::prepare(&system, &input, "energy").unwrap();
        let operators = crate::integrals::dipole_matrices(
            &prepared.setup.basis,
            Vec3::new(ORIGIN[0], ORIGIN[1], ORIGIN[2]),
        );
        let config = ResponseSolverConfig::default();
        let static_value = super::super::polarizability(
            &prepared.setup,
            &prepared.reference,
            prepared.target.as_ref(),
            prepared.target_vv10,
            &prepared.problem,
            &prepared.tensor,
            prepared.coefficients,
            &operators,
            &config,
        )
        .unwrap();
        let at_zero = solve(&prepared, &operators, Complex::ZERO, &config).unwrap();
        let diagnostics = &at_zero.diagnostics;
        for entry in diagnostics {
            for name in ["frame_inverse_defects", "frame_projector_defects"] {
                let defects = entry.get(name).unwrap().numeric_array().unwrap().1;
                assert!(defects.iter().all(|&value| value < 2e-10));
            }
        }
        let report = |feature: &str, values: &Value| {
            run(feature, &system, &Controls::from_value(values).unwrap()).unwrap()
        };
        let positive_frequency = 0.037;
        let damping = 0.006;
        let positive = report(
            "dynamic-polarizability",
            &controls
                .clone()
                .with("frequency", positive_frequency)
                .with("frequency_damping", damping),
        );
        let negative = report(
            "dynamic-polarizability",
            &controls
                .with("frequency", -positive_frequency)
                .with("frequency_damping", damping),
        );
        for (name, sign) in [("alpha_real", 1.0), ("alpha_imag", -1.0)] {
            let a = positive.get(name).unwrap().numeric_array().unwrap().1;
            let b = negative.get(name).unwrap().numeric_array().unwrap().1;
            assert!(a
                .into_iter()
                .zip(b)
                .all(|(a, b)| (a - sign * b).abs() < 1e-8));
        }
        let mut defect = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                for (actual, expected) in [
                    (at_zero.response.tensor[a][b], static_value.tensor[a][b]),
                    (at_zero.parent[a][b], static_value.parent[a][b]),
                    (at_zero.target[a][b], static_value.target[a][b]),
                    (at_zero.pt2[a][b], static_value.pt2[a][b]),
                    (at_zero.channels[0][a][b], static_value.channels[0][a][b]),
                    (at_zero.channels[1][a][b], static_value.channels[1][a][b]),
                    (at_zero.channels[2][a][b], static_value.channels[2][a][b]),
                ] {
                    defect = defect.max((actual - Complex::from(expected)).abs());
                }
            }
        }
        println!("Assembled dynamic PT2 exact static limit unrestricted={unrestricted} defect={defect:.17e}");
        assert!(defect < 1e-8);
        println!("Assembled dynamic PT2 z={positive_frequency}+{damping}i unrestricted={unrestricted} {}", positive.to_json());
    }
    std::fs::remove_file(basis).unwrap();
}

#[test]
fn causal_pt2_defaults_tighten_the_parent_and_preserve_an_explicit_loose_threshold() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
    );
    let options = Value::object()
        .with("basis", "sto-3g")
        .with("xc", "hf")
        .with("dispersion", "none")
        .with("pt2_same_spin", 0.28)
        .with("pt2_opposite_spin", 0.49)
        .with("frequency", 0.043)
        .with("frequency_damping", 0.007);
    let response = |input: &Value| {
        run(
            "dynamic-polarizability",
            &system,
            &Controls::from_value(input).unwrap(),
        )
    };
    let default = response(&options).unwrap();
    assert!(
        default
            .get("response_reference_residual")
            .unwrap()
            .as_f64()
            .unwrap()
            < 1e-10
    );
    let loose = options
        .with("conv_tol", 1.0)
        .with("conv_tol_dp", 0.25)
        .with("conv_tol_grad", 0.2);
    let error = response(&loose).unwrap_err();
    assert!(matches!(
        error,
        DftError::InvalidInput(message)
            if message.contains("PT2 frequency-response parent orbital residual")
    ));
}
