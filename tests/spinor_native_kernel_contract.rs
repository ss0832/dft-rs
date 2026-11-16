// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All active entries of a static projected descriptor Hessian. This is not
//! molecular nuclear, spatial-grid or higher local tensor acceptance.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::periodic::spinor_xc::{
    spinor_semilocal_point, spinor_semilocal_point_kernel, spinor_semilocal_point_taylor,
    SpinorXcPoint,
};
use dft_rs::xc::{resolve, FunctionalSpec};
fn point() -> [f64; 32] {
    let mut x = std::array::from_fn(|i| 0.012 * ((i * 7 + 2) as f64).sin());
    x[..4].copy_from_slice(&[0.225, 0.03, -0.02, 0.04]);
    x[16..20].copy_from_slice(&[2.0, 0.08, -0.07, 0.11]);
    x
}
#[test]
fn native_projected_kernel_matches_complete_taylor_columns() {
    let x = point();
    for name in ["svwn5", "pbe", "tpss"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        for rule in [4, 8] {
            let native =
                spinor_semilocal_point_kernel(&f, &SpinorXcPoint::from_descriptors(x), rule)
                    .unwrap()
                    .expect("smooth interior");
            assert_eq!(native.angular_order, rule);
            let mut errors = [0.0_f64; 3];
            for column in 0..native.active {
                let path = std::array::from_fn(|i| vec![x[i], if i == column { 1.0 } else { 0.0 }]);
                let old = spinor_semilocal_point_taylor(&f, &path, rule).unwrap();
                errors[0] = errors[0].max((native.energy - old.energy[0]).abs());
                for row in 0..32 {
                    errors[1] = errors[1].max((native.gradient[row] - old.gradient[row][0]).abs());
                    let value = if row < native.active {
                        native.kernel[row * native.active + column]
                    } else {
                        0.0
                    };
                    errors[2] = errors[2].max((value - old.gradient[row][1]).abs());
                    if row >= native.active {
                        assert_eq!(native.gradient[row], 0.0);
                        assert_eq!(old.gradient[row][1], 0.0);
                    }
                }
            }
            println!("{name} rule{rule} complete Taylor parity E/V/K {errors:?}");
            assert!(errors[0] < 2e-12 && errors[1] < 2e-11 && errors[2] < 2e-10);
        }
    }
}
#[test]
fn every_native_projected_gradient_and_kernel_entry_has_four_step_fits() {
    let x = point();
    let scale = x.map(|v| v.abs().max(0.05));
    for name in ["svwn5", "pbe", "tpss"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let native = spinor_semilocal_point_kernel(&f, &SpinorXcPoint::from_descriptors(x), 4)
            .unwrap()
            .expect("smooth interior");
        let n = native.active;
        let mut analytic = Vec::new();
        let mut names = Vec::new();
        for i in 0..n {
            analytic.push(native.gradient[i] * scale[i]);
            names.push(format!("{name} native gradient{i}"));
        }
        for i in 0..n {
            for j in 0..n {
                analytic.push(native.kernel[i * n + j] * scale[i] * scale[j]);
                names.push(format!("{name} native kernel{i},{j}"));
            }
        }
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.2, 0.1, 0.05, 0.025], stencils::Stencil::Second),
                ([0.4, 0.2, 0.1, 0.05], stencils::Stencil::Second),
                ([0.1, 0.05, 0.025, 0.0125], stencils::Stencil::Second),
                ([0.2, 0.1, 0.05, 0.025], stencils::Stencil::Fourth),
                ([0.2, 0.15, 0.1, 0.075], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; analytic.len()];
                for (k, &h) in steps.iter().enumerate() {
                    for j in 0..n {
                        let at = |s: f64| {
                            let mut y = x;
                            y[j] += s * scale[j];
                            spinor_semilocal_point(&f, &SpinorXcPoint::from_descriptors(y), 4)
                                .unwrap()
                        };
                        let p = at(h);
                        let m = at(-h);
                        let extra = if stencil == stencils::Stencil::Fourth {
                            Some((at(2.0 * h), at(-2.0 * h)))
                        } else {
                            None
                        };
                        errors[j][k] = (stencil.first(
                            p.energy,
                            m.energy,
                            extra.as_ref().map_or(0.0, |(p, _)| p.energy),
                            extra.as_ref().map_or(0.0, |(_, m)| m.energy),
                            h,
                        ) - analytic[j])
                            .abs();
                        for i in 0..n {
                            let slot = n + i * n + j;
                            errors[slot][k] = (stencil.first(
                                p.gradient[i],
                                m.gradient[i],
                                extra.as_ref().map_or(0.0, |(p, _)| p.gradient[i]),
                                extra.as_ref().map_or(0.0, |(_, m)| m.gradient[i]),
                                h,
                            ) * scale[i]
                                - analytic[slot])
                                .abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("native projected kernel components", &failures);
    }
}
#[test]
fn native_projection_fallback_and_input_errors_remain_explicit() {
    let f = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let mut x = point();
    x[4..16].fill(0.0);
    assert!(
        spinor_semilocal_point_kernel(&f, &SpinorXcPoint::from_descriptors(x), 4)
            .unwrap()
            .is_none()
    );
    x[0] = 0.0;
    assert!(spinor_semilocal_point_kernel(&f, &SpinorXcPoint::from_descriptors(x), 4).is_err());
    x = point();
    x[0] = f64::NAN;
    assert!(spinor_semilocal_point_kernel(&f, &SpinorXcPoint::from_descriptors(x), 4).is_err());
    assert!(
        spinor_semilocal_point_kernel(&f, &SpinorXcPoint::from_descriptors(point()), 3).is_err()
    );
}
