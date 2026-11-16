// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Actual noncollinear angular convergence and retained unsuccessful rules.
use dft_rs::periodic::spinor_xc::{spinor_semilocal_grid, spinor_semilocal_grid_converged};
use dft_rs::periodic::BlochMatrix;
use dft_rs::xc::{resolve, FunctionalSpec};

#[test]
fn actual_energy_and_fock_determine_angular_convergence() {
    let mut phi = BlochMatrix::zeros(1, 2);
    phi.re[(0, 0)] = 1.0;
    phi.im[(0, 0)] = 0.1;
    phi.re[(0, 1)] = 0.7;
    phi.im[(0, 1)] = -0.2;
    let mut momentum = std::array::from_fn(|_| BlochMatrix::zeros(1, 2));
    for d in 0..3 {
        for a in 0..2 {
            momentum[d].re[(0, a)] = 0.1 * (1 + d + a) as f64;
            momentum[d].im[(0, a)] = 0.07 * ((d + 1) * (a + 1)) as f64;
        }
    }
    let mut p = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    p[0].re[(0, 0)] = 0.9;
    p[0].re[(1, 1)] = 0.8;
    p[1].re[(0, 0)] = 0.1;
    p[2].re[(1, 1)] = 0.12;
    p[3].re[(0, 0)] = 0.15;
    p[3].re[(1, 1)] = -0.1;
    for name in ["pbe", "tpss", "r2scan", "m06l"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let result =
            spinor_semilocal_grid_converged(&f, &[0.7], &[&phi], &[&momentum], &[p.clone()], 1e-11)
                .unwrap();
        let d = &result.3;
        println!(
            "spinor angular {name}: history={:?} energy={:.3e} Fock={:.3e} \
            Hermitian={:.3e} terms={:.3e}",
            d.angular_refinement_history,
            d.angular_energy_defect,
            d.angular_operator_defect,
            d.hermiticity_defect,
            d.component_energy_defect
        );
        assert!(
            d.converged && d.angular_energy_defect <= 1e-11 && d.angular_operator_defect <= 1e-11
        );
        assert!(d.angular_refinement_history.len() >= 2);
        let fixed_rule = spinor_semilocal_grid(
            &f,
            &[0.7],
            &[&phi],
            &[&momentum],
            &[p.clone()],
            d.angular_order,
        )
        .unwrap();
        assert_eq!(result.0, fixed_rule.0);
        assert_eq!(result.1, fixed_rule.1);
        assert_eq!(result.2, fixed_rule.2);
        assert!(
            d.angular_refinement_history[0][1] > 1e-11
                || d.angular_refinement_history[0][2] > 1e-11
        );
        let unresolved =
            spinor_semilocal_grid_converged(&f, &[0.7], &[&phi], &[&momentum], &[p.clone()], 1e-30)
                .unwrap()
                .3;
        assert!(
            !unresolved.converged
                && unresolved.angular_order == dft_rs::periodic::spinor_xc::MAX_SPIN_ANGULAR_ORDER
        );
        assert_eq!(unresolved.angular_refinement_history.len(), 5);
        println!(
            "spinor angular unresolved {name}: history={:?}",
            unresolved.angular_refinement_history
        );
    }
}

#[test]
fn exact_collinear_grid_retains_energy_fock_and_refinement_diagnostics() {
    let mut phi = BlochMatrix::zeros(2, 1);
    phi.re[(0, 0)] = 0.8;
    phi.re[(1, 0)] = 0.6;
    let momentum = std::array::from_fn(|a| {
        let mut value = BlochMatrix::zeros(2, 1);
        value.im[(0, 0)] = -0.1 * (a + 1) as f64;
        value.im[(1, 0)] = 0.07 * (a + 1) as f64;
        value
    });
    let mut density = std::array::from_fn(|_| BlochMatrix::zeros(1, 1));
    density[0].re[(0, 0)] = 1.2;
    density[3].re[(0, 0)] = 0.3;
    let f = resolve(&FunctionalSpec::Combined("wb97m-v".into())).unwrap();
    let result = spinor_semilocal_grid_converged(
        &f,
        &[0.2, 0.4],
        &[&phi],
        &[&momentum],
        &[density.clone()],
        1e-11,
    )
    .unwrap();
    let fixed_rule =
        spinor_semilocal_grid(&f, &[0.2, 0.4], &[&phi], &[&momentum], &[density], 8).unwrap();
    assert_eq!(result.0, fixed_rule.0);
    assert_eq!(result.1, fixed_rule.1);
    assert_eq!(result.2, fixed_rule.2);
    assert!(result.3.converged);
    assert_eq!(result.3.angular_refinement_history, [[8.0, 0.0, 0.0]]);
}

#[test]
fn imaginary_rotated_seed_roundoff_stays_on_the_common_axis() {
    let mut phi = BlochMatrix::zeros(2, 2);
    phi.re[(0, 0)] = 0.8;
    phi.re[(0, 1)] = 0.6;
    phi.re[(1, 0)] = 0.4;
    phi.re[(1, 1)] = 0.9;
    let momentum = std::array::from_fn(|axis| {
        let mut p = BlochMatrix::zeros(2, 2);
        p.im[(0, 0)] = -0.14 * (axis + 1) as f64;
        p.im[(0, 1)] = -0.11 * (axis + 2) as f64;
        p.im[(1, 0)] = 0.04 * (axis + 3) as f64;
        p.im[(1, 1)] = 0.09 * (axis + 1) as f64;
        p
    });
    let mut density = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    let axis = [0.36, 0.48, 0.8];
    for (i, j, value) in [(0, 0, 0.8), (1, 1, 0.7), (0, 1, 0.2), (1, 0, 0.2)] {
        density[0].re[(i, j)] = value;
        for (spin, &component) in axis.iter().enumerate() {
            density[spin + 1].re[(i, j)] = 0.6 * value * component;
        }
    }
    let xc = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let integrate = |density| {
        spinor_semilocal_grid_converged(&xc, &[0.2, 0.5], &[&phi], &[&momentum], &[density], 1e-11)
            .unwrap()
    };
    let exact = integrate(density.clone());
    for (block, noise) in density.iter_mut().zip([0.0, -2.4e-16, 2.1e-16, -6.9e-18]) {
        block.im[(0, 1)] = noise;
        block.im[(1, 0)] = -noise;
    }
    let perturbed = integrate(density.clone());
    assert_eq!(perturbed.3.angular_refinement_history, [[8.0, 0.0, 0.0]]);
    assert!((perturbed.0 - exact.0).abs() < 1e-14);
    assert!(perturbed.1[0].re.max_abs_diff(&exact.1[0].re) < 1e-14);
    assert!(perturbed.1[0].im.max_abs_diff(&exact.1[0].im) < 1e-14);

    // A resolved transverse component still invokes the angular quadrature.
    density[2].re[(0, 1)] += 0.02;
    density[2].re[(1, 0)] += 0.02;
    let noncollinear = integrate(density);
    assert!(noncollinear.3.converged && noncollinear.3.angular_refinement_history.len() >= 2);
}
