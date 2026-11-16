// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! A common geometry walk must preserve accepted state identity and errors.
use dft_rs::math::Vec3;
use dft_rs::optimizer::{optimize_with_evaluator, OptOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::DftError;

fn molecule(x: f64) -> Molecule {
    Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(x, 0.2, -0.1),
    }])
}

#[test]
fn generic_analytic_walk_preserves_the_state_of_the_accepted_geometry() {
    let options = OptOptions {
        gtol: 1e-9,
        max_steps: 100,
        ..Default::default()
    };
    let value = optimize_with_evaluator(&molecule(0.8), &options, |m| {
        let p = m.atoms[0].position;
        let g = Vec3::new(p.x - 0.1, 2.0 * (p.y + 0.3), 3.0 * (p.z - 0.4));
        let e = 0.5 * (p.x - 0.1).powi(2) + (p.y + 0.3).powi(2) + 1.5 * (p.z - 0.4).powi(2);
        Ok((e, vec![g], p.to_array()))
    })
    .unwrap();
    assert!(value.converged);
    assert!(value.energy < 1e-16);
    assert_eq!(value.scf, value.molecule.atoms[0].position.to_array());
    assert!(value.trajectory.last().unwrap().grad_max < options.gtol);
    assert!(value
        .trajectory
        .windows(2)
        .all(|p| p[1].energy <= p[0].energy + 1e-12));
}

#[test]
fn only_scf_convergence_failures_are_recoverable_geometry_trials() {
    let mut rejected = 0;
    let value = optimize_with_evaluator(&molecule(0.8), &OptOptions::default(), |m| {
        let x = m.atoms[0].position.x;
        if x < 0.6 {
            rejected += 1;
            return Err(DftError::ScfNotConverged {
                iterations: 20,
                error: 0.01,
            });
        }
        Ok((
            10.0 * (x - 0.7).powi(2),
            vec![Vec3::new(20.0 * (x - 0.7), 0., 0.)],
            x,
        ))
    })
    .unwrap();
    assert!(rejected > 0 && value.converged);
    assert!((value.scf - 0.7).abs() < 1e-6);

    let mut calls = 0;
    let error = optimize_with_evaluator(&molecule(0.8), &OptOptions::default(), |_| {
        calls += 1;
        if calls > 1 {
            return Err(DftError::Unsupported {
                feature: "test geometry transport".into(),
                missing: "an analytic map".into(),
            });
        }
        Ok((1., vec![Vec3::new(1., 0., 0.)], ()))
    })
    .err()
    .unwrap();
    assert!(matches!(error, DftError::Unsupported { .. }));
    assert_eq!(calls, 2);
}

#[test]
fn malformed_evaluator_outputs_cannot_look_like_converged_geometries() {
    for (energy, gradient) in [
        (f64::NAN, vec![Vec3::zero()]),
        (0., vec![]),
        (0., vec![Vec3::new(f64::INFINITY, 0., 0.)]),
    ] {
        let result = optimize_with_evaluator(&molecule(0.8), &OptOptions::default(), |_| {
            Ok((energy, gradient.clone(), ()))
        });
        assert!(matches!(result, Err(DftError::InvalidInput(_))));
    }
    let options = OptOptions {
        max_steps: 1,
        gtol: 1e-14,
        ..Default::default()
    };
    let result = optimize_with_evaluator(&molecule(0.8), &options, |m| {
        let x = m.atoms[0].position.x;
        Ok((0.5 * x * x, vec![Vec3::new(x, 0., 0.)], ()))
    })
    .unwrap();
    assert!(!result.converged && result.iterations == 1);
}

#[test]
fn analytic_inverse_curvature_resolves_a_coupled_soft_mode_in_one_step() {
    use dft_rs::linalg::Matrix;
    use dft_rs::optimizer::optimize_with_inverse_hessian;
    // Rotate a soft and stiff Cartesian direction: a diagonal-only inverse
    // cannot remove this coupling. The exact inverse gives the Newton step.
    let h = [[1.0005, -0.9995, 0.], [-0.9995, 1.0005, 0.], [0., 0., 8.]];
    let mut inverse = Matrix::zeros(3, 3);
    inverse[(0, 0)] = 500.25;
    inverse[(1, 1)] = 500.25;
    inverse[(0, 1)] = 499.75;
    inverse[(1, 0)] = 499.75;
    inverse[(2, 2)] = 0.125;
    let evaluate = |m: &Molecule| {
        let x = m.atoms[0].position.to_array();
        let g: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| h[i][j] * x[j]).sum());
        let energy = 0.5 * (0..3).map(|i| x[i] * g[i]).sum::<f64>();
        Ok((energy, vec![Vec3::new(g[0], g[1], g[2])], x))
    };
    let opts = OptOptions {
        gtol: 1e-10,
        step_max: 0.3,
        max_steps: 100,
        ..Default::default()
    };
    let initial = molecule(0.1);
    let ordinary = optimize_with_evaluator(&initial, &opts, evaluate).unwrap();
    let preconditioned =
        optimize_with_inverse_hessian(&initial, &opts, &inverse, evaluate).unwrap();
    assert!(ordinary.converged && preconditioned.converged);
    assert_eq!(preconditioned.iterations, 1);
    assert!(ordinary.iterations > preconditioned.iterations);
    assert!(preconditioned.energy < 1e-22);
    assert_eq!(
        preconditioned.scf,
        preconditioned.molecule.atoms[0].position.to_array()
    );
    println!(
        "Coupled quadratic iterations: ordinary={} preconditioned={}",
        ordinary.iterations, preconditioned.iterations
    );
}

#[test]
fn an_indefinite_preconditioner_cannot_change_the_geometry_walk() {
    use dft_rs::linalg::Matrix;
    use dft_rs::optimizer::optimize_with_inverse_hessian;
    for dimension in [2, 3] {
        let mut metric = Matrix::identity(dimension);
        metric[(0, 0)] = -1.;
        let result =
            optimize_with_inverse_hessian(&molecule(0.2), &OptOptions::default(), &metric, |_| {
                panic!("invalid curvature must be rejected before evaluating the molecule")
            });
        assert!(matches!(
            result,
            Err::<dft_rs::optimizer::OptResult<()>, _>(DftError::InvalidInput(_))
        ));
    }
}
