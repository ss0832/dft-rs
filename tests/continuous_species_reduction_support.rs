// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Exact algebraic identities; no fictitious finite-difference convergence fit.
use dft_rs::linalg::{self, Matrix};
use dft_rs::pseudo_sub::reduction_support::{OrthogonalReducedChart, SupportProjection};

fn rotate(theta: f64) -> Matrix {
    let mut r = Matrix::zeros(2, 2);
    r[(0, 0)] = theta.cos();
    r[(0, 1)] = -theta.sin();
    r[(1, 0)] = theta.sin();
    r[(1, 1)] = theta.cos();
    r
}
fn fixture() -> (Matrix, Matrix, Matrix, Matrix, Matrix) {
    let diagonal: [f64; 4] = [0.7, 1.1, 1.7, 2.3];
    let mut s = Matrix::zeros(4, 4);
    let mut root = Matrix::zeros(4, 4);
    let mut c = Matrix::zeros(4, 4);
    for i in 0..4 {
        s[(i, i)] = diagonal[i];
        root[(i, i)] = diagonal[i].sqrt();
        c[(i, i)] = 1.0 / diagonal[i].sqrt();
    }
    let mut h = Matrix::zeros(4, 4);
    for i in 0..4 {
        for j in i..4 {
            let value = if i == j {
                -0.3 - 0.7 * i as f64
            } else {
                0.07 * (1 + i + j) as f64
            };
            h[(i, j)] = value;
            h[(j, i)] = value;
        }
    }
    let f = root.matmul_seq(&h).matmul_seq(&root);
    let mut y = Matrix::zeros(4, 2);
    let mut b = Matrix::zeros(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            y[(i, j)] = c[(i, j)];
            b[(i, j)] = c[(i, j + 2)];
        }
    }
    (y, b, f, s, c)
}
fn operator(phase: f64) -> Matrix {
    let mut a = Matrix::zeros(4, 4);
    for i in 0..4 {
        for j in i..4 {
            let x = ((i + 1) as f64 + phase * (j + 1) as f64).sin();
            a[(i, j)] = x;
            a[(j, i)] = x;
        }
    }
    a
}
fn squared(a: &Matrix) -> f64 {
    a.frobenius_dot(a)
}
fn subtract(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = a.clone();
    for (x, y) in c.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x -= y;
    }
    c
}

#[test]
fn independent_subspace_gauges_preserve_projection_gram_and_residual() {
    let (y, b, f, s, c) = fixture();
    let base = SupportProjection::new(&y, &b, &f, &s, 2.0).unwrap();
    let ry = rotate(0.37);
    let rb = rotate(-0.29);
    let rotated =
        SupportProjection::new(&y.matmul_seq(&ry), &b.matmul_seq(&rb), &f, &s, 2.0).unwrap();
    let mut gauge = Matrix::zeros(4, 4);
    for i in 0..2 {
        for j in 0..2 {
            gauge[(i, j)] = ry[(i, j)];
            gauge[(i + 2, j + 2)] = rb[(i, j)];
        }
    }
    assert!(base.orthonormality_error < 1e-14);
    assert!(rotated.orthonormality_error < 1e-14);
    assert!(base.normalized_pauli_identity_error < 1e-13);
    assert!(rotated.normalized_pauli_identity_error < 1e-13);
    assert_eq!(base.energy_scale_hartree, 1.0);
    assert_eq!(
        (base.physical_support_dimension, base.frozen_dimension),
        (2, 2)
    );
    let targets = [operator(0.13), operator(0.47), operator(-0.31)];
    let weighted: Vec<_> = targets
        .iter()
        .map(|a| base.weighted_operator(a).unwrap())
        .collect();
    let changed: Vec<_> = targets
        .iter()
        .map(|a| rotated.weighted_operator(a).unwrap())
        .collect();
    for i in 0..targets.len() {
        let expected = gauge.transpose_matmul_seq(&weighted[i].matmul_seq(&gauge));
        assert!(changed[i].max_abs_diff(&expected) < 1e-13);
        for j in 0..targets.len() {
            let error = (weighted[i].frobenius_dot(&weighted[j])
                - changed[i].frobenius_dot(&changed[j]))
            .abs();
            println!("support gauge Gram[{i},{j}] raw_error={error:.16e}");
            assert!(error < 1e-12);
        }
    }
    let residual = (squared(&subtract(&weighted[0], &weighted[1]))
        - squared(&subtract(&changed[0], &changed[1])))
    .abs();
    assert!(residual < 1e-12);
    // The old uniform full-AO metric has the same unitary-gauge identity.
    for target in &targets {
        let first = c.transpose_matmul_seq(&target.matmul_seq(&c));
        let second = rotated.transform_operator(target).unwrap();
        let error = (squared(&first) - squared(&second)).abs();
        println!("uniform fullAO gauge raw_error={error:.16e}");
        assert!(error < 1e-12);
    }
    assert_eq!(base.transform_dictionary(&targets).unwrap().len(), 3);
    println!("support normalized condition={} weight_condition={} orthogonality={} PauliIdentity={} rawEigenvalues={:?}",base.pauli_condition_number,base.weight_condition_number,base.orthonormality_error,base.normalized_pauli_identity_error,base.raw_pauli_eigenvalues);
}

#[test]
fn finite_reference_pauli_frozen_block_normalizes_to_identity() {
    let (y, b, f, s, _) = fixture();
    let base = SupportProjection::new(&y, &b, &f, &s, 2.0).unwrap();
    let sb = s.matmul_seq(&b);
    let mut q = Matrix::identity(4);
    let projector = b.matmul_transpose_seq(&sb);
    for (x, p) in q.as_mut_slice().iter_mut().zip(projector.as_slice()) {
        *x -= p;
    }
    let mut pauli = q.transpose_matmul_seq(&f.matmul_seq(&q));
    for (x, v) in pauli.as_mut_slice().iter_mut().zip(f.as_slice()) {
        *x -= v;
    }
    let barrier = sb.matmul_transpose_seq(&sb);
    for (x, v) in pauli.as_mut_slice().iter_mut().zip(barrier.as_slice()) {
        *x += 2.0 * v;
    }
    let normalized = base.weighted_operator(&pauli).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let error = (normalized[(i + 2, j + 2)] - expected).abs();
            println!("normalized PauliBB[{i},{j}] raw_error={error:.16e}");
            assert!(error < 1e-13);
        }
    }
}

#[test]
fn singular_indefinite_and_bad_spaces_are_explicit_errors() {
    let mut y = Matrix::zeros(3, 1);
    y[(0, 0)] = 1.0;
    let mut b = Matrix::zeros(3, 2);
    b[(1, 0)] = 1.0;
    b[(2, 1)] = 1.0;
    let s = Matrix::identity(3);
    let mut f = Matrix::zeros(3, 3);
    f[(1, 1)] = 2.0;
    assert!(SupportProjection::new(&y, &b, &f, &s, 2.0).is_err());
    f[(1, 1)] = 3.0;
    assert!(SupportProjection::new(&y, &b, &f, &s, 2.0).is_err());
    f[(1, 1)] = -2.0;
    let good = SupportProjection::new(&y, &b, &f, &s, 2.0).unwrap();
    assert!(good.raw_pauli_eigenvalues.iter().all(|x| *x > 0.0));
    y[(1, 0)] = 0.1;
    assert!(SupportProjection::new(&y, &b, &f, &s, 2.0).is_err());
    assert!(good.transform_operator(&Matrix::zeros(2, 2)).is_err());
    let mut nonhermitian = Matrix::zeros(3, 3);
    nonhermitian[(0, 1)] = 1.0;
    assert!(good.transform_operator(&nonhermitian).is_err());
    assert!(SupportProjection::new(&y, &b, &f, &s, f64::NAN).is_err());
}

#[test]
fn reduced_chart_preserves_metric_density_and_operator_energy() {
    let (_, _, f, s, _) = fixture();
    // Define a nonorthogonal two-function raw basis directly, both with a
    // finite frozen component.
    let mut raw = Matrix::zeros(4, 2);
    raw[(0, 0)] = 0.3;
    raw[(1, 0)] = 0.9;
    raw[(2, 0)] = -0.2;
    raw[(0, 1)] = -0.1;
    raw[(1, 1)] = 0.2;
    raw[(2, 1)] = 0.7;
    raw[(3, 1)] = 0.4;
    let cross = s.matmul_seq(&raw);
    let sr = raw.transpose_matmul_seq(&cross);
    let mut frozen = Matrix::zeros(4, 1);
    frozen[(0, 0)] = 1.0 / s[(0, 0)].sqrt();
    let chart = OrthogonalReducedChart::new(&s, &cross, &sr, &frozen, 1e-10).unwrap();
    assert_eq!(chart.transform.cols, 2);
    assert!(chart.metric_error < 1e-13 && chart.frozen_overlap < 1e-13);
    let x = linalg::Orthogonalizer::canonical(&sr, 1e-12).unwrap().x;
    // Two distinct occupied endpoints check energy duality and round trips.
    for column in 0..2 {
        let mut density = Matrix::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                density[(i, j)] = 2.0 * x[(i, column)] * x[(j, column)];
            }
        }
        let lifted = chart.lift_density(&density).unwrap();
        let recovered = chart.pullback_density(&lifted).unwrap();
        let operator = chart.pullback_operator(&f).unwrap();
        let defect = density.frobenius_dot(&operator) - lifted.frobenius_dot(&f);
        println!(
            "raw chart endpoint={column} energy_defect={defect:.16e} metric={:.16e} frozen={:.16e}",
            chart.metric_error, chart.frozen_overlap
        );
        assert!(defect.abs() < 1e-13);
        assert!((lifted.frobenius_dot(&s) - 2.0).abs() < 1e-13);
        assert!(recovered.max_abs_diff(&density) < 1e-13);
        let squared = lifted.matmul_seq(&s).matmul_seq(&lifted);
        for (a, b) in squared.as_slice().iter().zip(lifted.as_slice()) {
            assert!((a - 2.0 * b).abs() < 1e-13);
        }
    }
}

#[test]
fn reduced_chart_is_invariant_to_frozen_orbital_rotation() {
    let s = Matrix::identity(5);
    let mut raw = Matrix::zeros(5, 2);
    raw[(0, 0)] = 0.3;
    raw[(1, 1)] = -0.4;
    raw[(2, 0)] = 1.0;
    raw[(3, 1)] = 0.8;
    raw[(4, 0)] = 0.2;
    let sr = raw.transpose_matmul_seq(&raw);
    let mut frozen = Matrix::zeros(5, 2);
    frozen[(0, 0)] = 1.0;
    frozen[(1, 1)] = 1.0;
    let base = OrthogonalReducedChart::new(&s, &raw, &sr, &frozen, 1e-10).unwrap();
    let rotated = frozen.matmul_seq(&rotate(0.37));
    let other = OrthogonalReducedChart::new(&s, &raw, &sr, &rotated, 1e-10).unwrap();
    assert!(base.transform.max_abs_diff(&other.transform) < 1e-13);
    let empty = Matrix::zeros(5, 0);
    let no_frozen = OrthogonalReducedChart::new(&s, &raw, &sr, &empty, 1e-10).unwrap();
    assert!(no_frozen.transform.max_abs_diff(&raw) < 1e-13);
}

#[test]
fn reduced_chart_refuses_missing_reference_span_or_endpoint_rank_loss() {
    let s = Matrix::identity(3);
    let sr = Matrix::identity(2);
    let mut cross = Matrix::zeros(3, 2);
    cross[(0, 0)] = 1.0;
    cross[(1, 1)] = 1.0;
    let mut frozen = Matrix::zeros(3, 1);
    frozen[(0, 0)] = 1.0;
    let error = OrthogonalReducedChart::new(&s, &cross, &sr, &frozen, 1e-10).unwrap_err();
    assert!(error.to_string().contains("common AO rank"));
    frozen[(0, 0)] = 0.0;
    frozen[(2, 0)] = 1.0;
    cross[(0, 0)] = 0.8;
    let error = OrthogonalReducedChart::new(&s, &cross, &sr, &frozen, 1e-10).unwrap_err();
    assert!(error.to_string().contains("entire common raw AO span"));
    cross[(0, 0)] = 1.0;
    let good = OrthogonalReducedChart::new(&s, &cross, &sr, &frozen, 1e-10).unwrap();
    assert!(good.pullback_operator(&Matrix::identity(2)).is_err());
    assert!(good.lift_density(&Matrix::identity(3)).is_err());
    assert!(good.pullback_density(&Matrix::identity(2)).is_err());
}
