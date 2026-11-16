// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every active descriptor gradient/kernel entry of the global spinor XC
//! Taylor provider against independently differentiated native potentials.
//! All 1552 active entries pass: p=1.937461..2.085495, minimum
//! R2=0.99989851. Two failed initial stages are retained and resolved.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::periodic::spinor_xc::{
    spinor_semilocal_point, spinor_semilocal_point_taylor, SpinorXcPoint,
};
use dft_rs::xc::{resolve, Family, FunctionalSpec};

fn point() -> [f64; 32] {
    SpinorXcPoint {
        density: [0.91, 0.12, 0.17, 0.23],
        gradient: [
            [0.19, -0.13, 0.17],
            [0.041, -0.027, 0.036],
            [-0.033, 0.051, 0.024],
            [0.057, 0.019, -0.043],
        ],
        tau: [0.83, 0.074, -0.061, 0.093],
        current: [
            [0.071, -0.053, 0.043],
            [-0.031, 0.024, 0.017],
            [0.023, -0.019, 0.037],
            [0.041, 0.029, -0.026],
        ],
    }
    .descriptors()
}

#[test]
fn all_semilocal_descriptor_kernel_components_pass_independent_fits() {
    let x = point();
    let mut failures = Vec::new();
    for name in ["svwn5", "pbe", "tpss", "pbe0"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let active = match functional.family {
            Family::Lda => 4,
            Family::Gga => 16,
            Family::MetaGga => 32,
        };
        let mut analytic = vec![[0.0; 32]; 32];
        let native =
            spinor_semilocal_point(&functional, &SpinorXcPoint::from_descriptors(x), 8).unwrap();
        for column in 0..32 {
            let path = std::array::from_fn(|i| vec![x[i], if i == column { 1.0 } else { 0.0 }]);
            let output = spinor_semilocal_point_taylor(&functional, &path, 8).unwrap();
            assert!((output.energy[0] - native.energy).abs() < 1e-12);
            assert!((output.energy[1] - native.gradient[column]).abs() < 1e-11);
            assert_eq!(output.angular_order, 8);
            for row in 0..32 {
                assert!((output.gradient[row][0] - native.gradient[row]).abs() < 1e-11);
                analytic[row][column] = output.gradient[row][1];
                if row >= active || column >= active {
                    assert_eq!(analytic[row][column], 0.0);
                }
            }
        }
        let names: Vec<_> = (0..active)
            .flat_map(|row| {
                (0..active).map(move |column| format!("{name} spinor XC kernel ({row},{column})"))
            })
            .collect();
        let stages = [
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.008, 0.006, 0.004, 0.003], stencils::Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
        ];
        let errors_at = |steps: &[f64; 4], stencil: stencils::Stencil| {
            let mut errors = vec![[0.0; 4]; active * active];
            for (k, &h) in steps.iter().enumerate() {
                for column in 0..active {
                    let sample = |shift: f64| {
                        let mut moved = x;
                        moved[column] += shift;
                        spinor_semilocal_point(
                            &functional,
                            &SpinorXcPoint::from_descriptors(moved),
                            8,
                        )
                        .unwrap()
                        .gradient
                    };
                    let plus = sample(h);
                    let minus = sample(-h);
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (sample(2.0 * h), sample(-2.0 * h))
                    } else {
                        ([0.0; 32], [0.0; 32])
                    };
                    for row in 0..active {
                        errors[row * active + column][k] =
                            (stencil.first(plus[row], minus[row], plus2[row], minus2[row], h)
                                - analytic[row][column])
                                .abs();
                    }
                }
            }
            if errors[0].iter().all(|v| *v > 0.0) {
                println!(
                    "{name} kernel log RMSE={:.3e}",
                    convergence::error_power_fit(steps, &errors[0]).log_rmse
                );
            }
            errors
        };
        let initial = errors_at(&stages[0].0, stages[0].1);
        let mut failed = Vec::new();
        components::fit_components(&names, &[stages[0].0], |_| initial.clone(), &mut failed);
        if !failed.is_empty() {
            let indices: Vec<_> = failed
                .iter()
                .map(|name| names.iter().position(|n| n == name).unwrap())
                .collect();
            let selected: Vec<_> = indices.iter().map(|i| names[*i].clone()).collect();
            stencils::fit_components_staged(
                &selected,
                &stages[1..],
                |h, stencil| {
                    let measured = errors_at(h, stencil);
                    indices.iter().map(|i| measured[*i]).collect()
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("spinor XC mixed Taylor kernels", &failures);
}
