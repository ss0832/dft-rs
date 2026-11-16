// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent closed-block complex oracle for the no-pair operator transform.
use dft_rs::properties::no_pair::no_pair_operator_taylor;
#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::periodic::BlochMatrix;
type J = Taylor<5>;
type Z = (J, J);
type M = [[Z; 4]; 4];

#[test]
fn common_driver_supplied_dimensionless_operator_coefficients_and_contracts() {
    use dft_rs::driver::{run, Controls, System, ToValue, Value};
    let (h, s, o, _) = fixture(J::from_coefficients(&[0.0, 1.0]));
    let request = Value::object()
        .with("response_order", 2usize)
        .with("spectrum_separator", 0.0)
        .with("dirac_hamiltonian_real", h[0].re.clone())
        .with("dirac_hamiltonian_imaginary", h[0].im.clone())
        .with("dirac_hamiltonian_direction_real", h[1].re.clone())
        .with("dirac_hamiltonian_direction_imaginary", h[1].im.clone())
        .with("dirac_overlap_real", s[0].re.clone())
        .with("dirac_overlap_imaginary", s[0].im.clone())
        .with("dirac_overlap_direction_real", s[1].re.clone())
        .with("dirac_overlap_direction_imaginary", s[1].im.clone());
    let mut supplied = request.clone();
    supplied.set(
        "dirac_operator_real",
        o[..3].iter().map(|a| a.re.clone()).collect::<Vec<_>>(),
    );
    supplied.set(
        "dirac_operator_imaginary",
        o[..3].iter().map(|a| a.im.clone()).collect::<Vec<_>>(),
    );
    let run_request = |value: &Value| {
        run(
            "no-pair-projector",
            &System::default(),
            &Controls::from_value(value).unwrap(),
        )
    };
    let report = run_request(&supplied).unwrap();
    let expected = no_pair_operator_taylor(&h[..2], &s[..2], &o[..3], 2, 0.0, 1e-12).unwrap();
    assert_eq!(
        report.get("projected_operator_coefficients_real").unwrap(),
        &expected
            .operator
            .iter()
            .map(|a| a.re.clone())
            .collect::<Vec<_>>()
            .to_value()
    );
    assert_eq!(
        report
            .get("projected_operator_coefficients_imaginary")
            .unwrap(),
        &expected
            .operator
            .iter()
            .map(|a| a.im.clone())
            .collect::<Vec<_>>()
            .to_value()
    );
    assert_eq!(
        report
            .get("units")
            .unwrap()
            .get("projected_operator_coefficients_real"),
        Some(&"dimensionless".to_value())
    );
    assert!(report.get("projected_operator_model").is_some());
    assert!(run_request(&request)
        .unwrap()
        .get("projected_operator_coefficients_real")
        .is_none());
    let mut real_only = request.clone();
    real_only.set("dirac_operator_real", vec![o[0].re.clone()]);
    assert!(run_request(&real_only).is_ok());
    let mut imaginary_only = request.clone();
    imaginary_only.set("dirac_operator_imaginary", vec![o[0].im.clone()]);
    assert!(run_request(&imaginary_only).is_ok());
    for bad in [
        Value::Array(vec![]),
        o.iter()
            .map(|a| a.re.clone())
            .collect::<Vec<_>>()
            .to_value(),
        vec![vec![vec![0.0]]].to_value(),
        vec![vec![vec![f64::NAN; 4]; 4]].to_value(),
        o[0].re.to_value(),
    ] {
        let mut invalid = request.clone();
        invalid.set("dirac_operator_real", bad);
        assert!(run_request(&invalid).is_err());
    }
}

fn z(x: f64) -> Z {
    (J::from_f64(x), J::from_f64(0.0))
}
fn add(a: Z, b: Z) -> Z {
    (a.0 + b.0, a.1 + b.1)
}
fn mul(a: Z, b: Z) -> Z {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn adj(a: &M) -> M {
    std::array::from_fn(|i| std::array::from_fn(|j| (a[j][i].0, -a[j][i].1)))
}
fn mm(a: &M, b: &M) -> M {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..4).fold(z(0.0), |v, k| add(v, mul(a[i][k], b[k][j]))))
    })
}
fn series(a: &M) -> Vec<BlochMatrix> {
    (0..5)
        .map(|k| {
            let mut b = BlochMatrix::zeros(4, 4);
            for i in 0..4 {
                for j in 0..4 {
                    b.re[(i, j)] = a[i][j].0.c[k];
                    b.im[(i, j)] = a[i][j].1.c[k];
                }
            }
            b
        })
        .collect()
}
fn fixture(
    t: J,
) -> (
    Vec<BlochMatrix>,
    Vec<BlochMatrix>,
    Vec<BlochMatrix>,
    Vec<BlochMatrix>,
) {
    let j = J::from_f64;
    let mut h = [[z(0.0); 4]; 4];
    let mut p = h;
    let mut o = h;
    // Two exact Hermitian 2x2 blocks: H^2=E^2 I, P+=(I+H/E)/2.
    // No eigensolver, transport or production projector enters this oracle.
    for b in 0..2 {
        let a = 2 * b;
        let d = a + 1;
        let m = j(1.7 + b as f64) + t.scale(0.13);
        let x = j(0.43 + 0.17 * b as f64) + t.scale(0.11);
        let y = j(0.29 + 0.09 * b as f64) - t.scale(0.07);
        let e = (m * m + x * x + y * y).sqrt();
        h[a][a] = (m, j(0.0));
        h[d][d] = (-m, j(0.0));
        h[a][d] = (x, y);
        h[d][a] = (x, -y);
        p[a][a] = ((j(1.0) + m / e).scale(0.5), j(0.0));
        p[d][d] = ((j(1.0) - m / e).scale(0.5), j(0.0));
        p[a][d] = ((x / e).scale(0.5), (y / e).scale(0.5));
        p[d][a] = ((x / e).scale(0.5), (-y / e).scale(0.5));
    }
    for a in 0..4 {
        for b in a..4 {
            let re = (t.scale(0.19 + 0.03 * (a + b) as f64) + j(0.17 * (1 + a + b) as f64)).exp();
            let im = if a == b {
                j(0.0)
            } else {
                (t.scale(0.23 + 0.02 * (a * b) as f64) + j(0.11 * (1 + a + 2 * b) as f64)).exp()
            };
            o[a][b] = (re, im);
            o[b][a] = (re, -im);
        }
    }
    let had = [
        [1., 1., 1., 1.],
        [1., -1., 1., -1.],
        [1., 1., -1., -1.],
        [1., -1., -1., 1.],
    ];
    let basis: M = std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let scale = (t.scale(0.08 + 0.02 * a as f64) + j(0.03 * a as f64))
                .exp()
                .scale(0.5 * had[a][b]);
            let phase = 0.31 * b as f64;
            (scale.scale(phase.cos()), scale.scale(phase.sin()))
        })
    });
    let bt = adj(&basis);
    let s = mm(&bt, &basis);
    let cov_h = mm(&mm(&bt, &h), &basis);
    let cov_o = mm(&mm(&bt, &o), &basis);
    let exact = mm(&mm(&bt, &mm(&mm(&p, &o), &p)), &basis);
    (series(&cov_h), series(&s), series(&cov_o), series(&exact))
}
fn entry(a: &BlochMatrix, i: usize, k: usize, imag: bool) -> f64 {
    if imag {
        a.im[(i, k)]
    } else {
        a.re[(i, k)]
    }
}
#[test]
fn covariant_hamiltonian_identity_hermiticity_and_input_contracts() {
    let (h, s, o, _) = fixture(J::from_coefficients(&[0.0, 1.0]));
    let projected_h = no_pair_operator_taylor(&h, &s, &h, 4, 0.0, 1e-12).unwrap();
    let identity = no_pair_operator_taylor(&h, &s, &s, 4, 0.0, 1e-12).unwrap();
    let property = no_pair_operator_taylor(&h, &s, &o, 4, 0.0, 1e-12).unwrap();
    // For O=S (the covariant identity), S P S P S = S P S by P S P=P.
    for n in 0..5 {
        for i in 0..4 {
            for k in 0..4 {
                for imag in [false, true] {
                    assert!(
                        (entry(&projected_h.operator[n], i, k, imag)
                            - entry(&projected_h.sector.projected_hamiltonian[n], i, k, imag))
                        .abs()
                            < 1e-10
                    );
                    let mut expected = (0.0, 0.0);
                    for a in 0..=n {
                        for b in 0..=n - a {
                            let c = n - a - b;
                            for r in 0..4 {
                                for v in 0..4 {
                                    let x = (s[a].re[(i, r)], s[a].im[(i, r)]);
                                    let y = (
                                        identity.sector.projector[b].re[(r, v)],
                                        identity.sector.projector[b].im[(r, v)],
                                    );
                                    let z = (s[c].re[(v, k)], s[c].im[(v, k)]);
                                    let xy = (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
                                    expected.0 += xy.0 * z.0 - xy.1 * z.1;
                                    expected.1 += xy.0 * z.1 + xy.1 * z.0;
                                }
                            }
                        }
                    }
                    assert!(
                        (entry(&identity.operator[n], i, k, imag)
                            - if imag { expected.1 } else { expected.0 })
                        .abs()
                            < 1e-10
                    );
                }
            }
        }
        assert!(property.operator[n].hermiticity_error() < 1e-10);
    }
    assert!(no_pair_operator_taylor(&h, &s, &[], 4, 0.0, 1e-12).is_err());
    let mut bad = o.clone();
    bad[0].re[(0, 1)] += 0.1;
    assert!(no_pair_operator_taylor(&h, &s, &bad, 4, 0.0, 1e-12).is_err());
    assert!(no_pair_operator_taylor(&h, &s, &o, 4, 100.0, 1e-12).is_err());
}
#[test]
fn every_independent_operator_and_metric_derivative_has_four_step_fit() {
    let (h, s, o, oracle) = fixture(J::from_coefficients(&[0.0, 1.0]));
    let actual = no_pair_operator_taylor(&h, &s, &o, 4, 0.0, 1e-12).unwrap();
    assert_eq!(actual.sector.electronic_dimension, 2);
    for n in 0..5 {
        for i in 0..4 {
            for k in 0..4 {
                for imag in [false, true] {
                    assert!(
                        (entry(&actual.operator[n], i, k, imag) - entry(&oracle[n], i, k, imag))
                            .abs()
                            < 1e-10
                    );
                }
            }
        }
    }
    let steps = [0.08, 0.04, 0.02, 0.01];
    let mut failed = Vec::new();
    for which in 0..2 {
        for n in 1..=4 {
            for i in 0..4 {
                for k in i..4 {
                    for imag in [false, true] {
                        if imag && i == k {
                            continue;
                        } // Hermitian diagonal imaginary entries are exact structural zeros.
                        let target = entry(
                            if which == 0 {
                                &actual.operator[n]
                            } else {
                                &s[n]
                            },
                            i,
                            k,
                            imag,
                        ) * n as f64;
                        let errors = steps.map(|h| {
                            let (_, sp, _, qp) = fixture(J::from_coefficients(&[h, 1.0]));
                            let (_, sm, _, qm) = fixture(J::from_coefficients(&[-h, 1.0]));
                            let plus = if which == 0 { &qp } else { &sp };
                            let minus = if which == 0 { &qm } else { &sm };
                            ((entry(&plus[n - 1], i, k, imag) - entry(&minus[n - 1], i, k, imag))
                                / (2.0 * h)
                                - target)
                                .abs()
                        });
                        let label = format!(
                            "{} order={n} ({i},{k}) imag={imag}",
                            if which == 0 { "operator" } else { "metric" }
                        );
                        if !components::component_fit_of_order(&label, &steps, &errors, 2.0) {
                            failed.push(label);
                        }
                    }
                }
            }
        }
    }
    components::assert_no_failures("no-pair operator and metric", &failed);
}
