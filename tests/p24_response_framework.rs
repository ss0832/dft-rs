// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::dual::Scalar;
use dft_rs::response::complex::Complex;
use dft_rs::response::krylov::{solve, LinearOperator, OperatorStructure, ResponseSolverConfig};
use dft_rs::response::taylor::Taylor;
use dft_rs::{DftError, Result};

struct Operator {
    entries: Vec<Vec<Complex>>,
    structure: OperatorStructure,
}

#[test]
fn scaled_sources_and_explicit_goldstone_constraints_preserve_the_operator() {
    use dft_rs::response::constraints::{solve_constrained, ConstraintSpace};
    for structure in [
        OperatorStructure::PositiveDefinite,
        OperatorStructure::Hermitian,
        OperatorStructure::General,
    ] {
        let op = Operator {
            entries: vec![vec![2.0.into(), 0.0.into()], vec![0.0.into(), 3.0.into()]],
            structure,
        };
        for scale in [1e-250, 1e250] {
            let rhs = vec![Complex::from(2.0 * scale), Complex::from(6.0 * scale)];
            let r = solve(&op, &rhs, None, &Default::default()).unwrap();
            assert!((r.amplitudes[0].re / scale - 1.0).abs() < 1e-10);
            assert!((r.amplitudes[1].re / scale - 2.0).abs() < 1e-10);
        }
    }
    let op = Operator {
        entries: vec![
            vec![0.0.into(), 0.0.into(), 0.0.into()],
            vec![0.0.into(), 2.0.into(), 0.0.into()],
            vec![0.0.into(), 0.0.into(), 3.0.into()],
        ],
        structure: OperatorStructure::Hermitian,
    };
    let space = ConstraintSpace::new(3, &[vec![1.0.into(), 0.0.into(), 0.0.into()]]).unwrap();
    let b = vec![2.0.into(), 4.0.into(), 6.0.into()];
    let r = solve_constrained(&op, &b, &space, &Default::default()).unwrap();
    assert!(r.constraint_defect < 1e-12);
    assert!((r.multipliers[0] - Complex::from(2.0)).abs() < 1e-12);
    assert!((r.solution.amplitudes[1] - Complex::from(2.0)).abs() < 1e-12);
    assert!((r.solution.amplitudes[2] - Complex::from(2.0)).abs() < 1e-12);
    assert!(ConstraintSpace::new(
        3,
        &[
            vec![1.0.into(), 0.0.into(), 0.0.into()],
            vec![2.0.into(), 0.0.into(), 0.0.into()]
        ]
    )
    .is_err());
}

#[test]
fn analytic_polarization_extracts_repeated_mixed_derivatives() {
    use dft_rs::response::mixed::polarized_mixed;
    let directions = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
    ];
    let mixed = polarized_mixed(2, &directions, |d, n| {
        assert_eq!(n, 5);
        Ok(vec![d[0].powi(2) * d[1].powi(3)])
    })
    .unwrap();
    assert!((mixed.value[0] - 12.0).abs() < 1e-12);
    assert_eq!(mixed.directional_evaluations, 11);
    let directions = vec![vec![0.1, 0.2], vec![0.3, -0.2], vec![-0.2, 0.4]];
    let mixed = polarized_mixed(2, &directions, |d, n| {
        let jets = [vec![0.7, d[0] + d[1], 0.0, 0.0]];
        Ok(vec![
            dft_rs::response::series::evaluate_series(&jets, |x| x[0].exp())?[n],
        ])
    })
    .unwrap();
    assert!((mixed.value[0] - 0.7_f64.exp() * 0.3 * 0.1 * 0.2).abs() < 1e-14);
}
impl LinearOperator for Operator {
    fn dimension(&self) -> usize {
        self.entries.len()
    }
    fn structure(&self) -> OperatorStructure {
        self.structure
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        Ok(self
            .entries
            .iter()
            .map(|row| {
                row.iter()
                    .zip(x)
                    .fold(Complex::ZERO, |s, (&a, &x)| s + a * x)
            })
            .collect())
    }
}

#[test]
fn krylov_solvers_solve_independent_complex_systems() {
    let expected = vec![
        Complex::new(0.3, -0.8),
        Complex::new(-0.2, 0.7),
        Complex::new(0.9, 0.1),
    ];
    for structure in [
        OperatorStructure::PositiveDefinite,
        OperatorStructure::Hermitian,
        OperatorStructure::General,
    ] {
        let a = match structure {
            OperatorStructure::PositiveDefinite => vec![
                vec![4.0.into(), Complex::new(0.2, 0.4), 0.1.into()],
                vec![Complex::new(0.2, -0.4), 3.0.into(), 0.5.into()],
                vec![0.1.into(), 0.5.into(), 2.0.into()],
            ],
            OperatorStructure::Hermitian => vec![
                vec![2.0.into(), Complex::new(0.5, 0.3), 0.0.into()],
                vec![Complex::new(0.5, -0.3), (-1.0).into(), 0.7.into()],
                vec![0.0.into(), 0.7.into(), 3.0.into()],
            ],
            OperatorStructure::General => vec![
                vec![
                    Complex::new(2.0, 0.1),
                    Complex::new(1.0, 0.5),
                    (-0.4).into(),
                ],
                vec![0.0.into(), Complex::new(-3.0, 0.4), 0.9.into()],
                vec![0.1.into(), 0.0.into(), Complex::new(1.2, -0.7)],
            ],
        };
        let op = Operator {
            entries: a,
            structure,
        };
        let b = op.apply(&expected).unwrap();
        let report = solve(&op, &b, None, &ResponseSolverConfig::default()).unwrap();
        for (x, y) in report.amplitudes.iter().zip(&expected) {
            assert!((*x - *y).abs() < 1e-10);
        }
        let fresh = op.apply(&report.amplitudes).unwrap();
        let residual = fresh
            .iter()
            .zip(&b)
            .fold(0.0_f64, |s, (&x, &y)| s.hypot((x - y).abs()));
        let norm = b.iter().fold(0.0_f64, |s, x| s.hypot(x.abs()));
        assert!((residual / norm - report.relative_residual).abs() < 1e-14);
    }
}

#[test]
fn gmres_restart_and_failure_contracts() {
    let op = Operator {
        entries: vec![vec![3.0.into(), 0.2.into()], vec![0.3.into(), 2.0.into()]],
        structure: OperatorStructure::General,
    };
    let cfg = ResponseSolverConfig {
        gmres_restart: 1,
        ..Default::default()
    };
    let b = vec![1.0.into(), (-0.4).into()];
    let result = solve(&op, &b, None, &cfg).unwrap();
    assert!(result.iterations > 2);
    assert!(result.relative_residual <= cfg.tolerance);
    let cfg = ResponseSolverConfig {
        max_iterations: 1,
        ..cfg
    };
    assert!(matches!(
        solve(&op, &b, None, &cfg),
        Err(DftError::ResponseNotConverged { .. })
    ));
    assert!(solve(
        &op,
        &[Complex::new(f64::NAN, 0.0), Complex::ONE],
        None,
        &cfg
    )
    .is_err());
    assert!(solve(&op, &[Complex::ONE], None, &cfg).is_err());
    assert!(solve(&op, &b, Some(&[1.0, -1.0]), &cfg).is_err());
    let zero = solve(&op, &[Complex::ZERO; 2], None, &cfg).unwrap();
    assert_eq!(zero.iterations, 0);
}

#[test]
fn sixth_order_mixed_taylor_coefficients_match_closed_forms() {
    type Inner = Taylor<f64, 7>;
    type Mixed = Taylor<Inner, 7>;
    let x = Mixed::constant(Inner::variable(0.3, 1.0));
    let y = Mixed::variable(Inner::constant(-0.2), Inner::constant(1.0));
    let result = (x + y).exp();
    let constant = 0.1_f64.exp();
    for a in 0..=6 {
        for b in 0..=6 {
            let mixed_derivative = result.derivative(a).derivative(b);
            assert!((mixed_derivative - constant).abs() < 1e-12);
        }
    }
    let x = Inner::variable(2.0, 1.0);
    let inverse = x.recip();
    let logarithm = x.ln();
    let mut factorial = 1.0;
    for n in 1..=6 {
        factorial *= n as f64;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        assert!(
            (inverse.derivative(n) - sign * factorial / 2.0_f64.powi(n as i32 + 1)).abs() < 1e-12
        );
        assert!(
            (logarithm.derivative(n) + sign * factorial / (n as f64 * 2.0_f64.powi(n as i32)))
                .abs()
                < 1e-12
        );
    }
}

#[test]
fn taylor_special_functions_obey_differential_identities() {
    type T = Taylor<f64, 9>;
    let x = T::variable(0.7, 1.0);
    let erf = x.erf();
    let rhs = (-x * x).exp() * T::from_f64(2.0 / std::f64::consts::PI.sqrt());
    let erfcx = x.erfcx();
    let erfcx_rhs = T::from_f64(2.0) * x * erfcx - T::from_f64(2.0 / std::f64::consts::PI.sqrt());
    let e1 = x.e1_scaled();
    let e1_rhs = e1 - x.recip();
    for n in 0..7 {
        assert!(((n + 1) as f64 * erf.coefficients[n + 1] - rhs.coefficients[n]).abs() < 1e-12);
        assert!(
            ((n + 1) as f64 * erfcx.coefficients[n + 1] - erfcx_rhs.coefficients[n]).abs() < 1e-12
        );
        assert!(((n + 1) as f64 * e1.coefficients[n + 1] - e1_rhs.coefficients[n]).abs() < 1e-12);
    }
    let negative = T::variable(-8.0, 1.0).cbrt();
    assert!((negative.derivative(1) - 1.0 / 12.0).abs() < 1e-14);
}
