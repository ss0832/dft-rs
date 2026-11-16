// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex as C;
use dft_rs::response::spinor_tda::{
    spinor_tda_eigenpair_taylor, spinor_tda_pole_taylor, SpinorTdaConfig,
};

fn values(x: f64, y: f64) -> [f64; 4] {
    [
        0.7 + 0.11 * x + 0.04 * y + 0.03 * x * y,
        2.1 - 0.07 * x + 0.09 * y + 0.02 * x * x,
        0.23 + 0.13 * x - 0.08 * y + 0.03 * x * y,
        0.17 - 0.06 * x + 0.12 * y + 0.025 * y * y,
    ]
}
fn matrix(v: [f64; 4]) -> BlochMatrix {
    let mut a = BlochMatrix::zeros(2, 2);
    a.re[(0, 0)] = v[0];
    a.re[(1, 1)] = v[1];
    a.re[(0, 1)] = v[2];
    a.re[(1, 0)] = v[2];
    a.im[(0, 1)] = v[3];
    a.im[(1, 0)] = -v[3];
    a
}
fn jets(dx: f64, dy: f64) -> Vec<BlochMatrix> {
    let (x, y) = (0.17, -0.23);
    vec![
        matrix(values(x, y)),
        matrix([
            0.11 * dx + 0.04 * dy + 0.03 * (dx * y + dy * x),
            -0.07 * dx + 0.09 * dy + 0.04 * x * dx,
            0.13 * dx - 0.08 * dy + 0.03 * (dx * y + dy * x),
            -0.06 * dx + 0.12 * dy + 0.05 * y * dy,
        ]),
        matrix([
            0.03 * dx * dy,
            0.02 * dx * dx,
            0.03 * dx * dy,
            0.025 * dy * dy,
        ]),
    ]
}
fn get(v: &BlochMatrix, i: usize) -> C {
    C::new(v.re[(i, 0)], v.im[(i, 0)])
}
fn flatten(v: &BlochMatrix) -> [f64; 4] {
    [v.re[(0, 0)], v.im[(0, 0)], v.re[(1, 0)], v.im[(1, 0)]]
}
fn exact(x: f64, y: f64, base: &BlochMatrix) -> [f64; 4] {
    let [a, d, re, im] = values(x, y);
    let energy = 0.5 * (a + d) - ((0.5 * (a - d)).powi(2) + re * re + im * im).sqrt();
    let norm = (re * re + im * im + (a - energy).powi(2)).sqrt();
    let raw = [
        C::new(-re / norm, -im / norm),
        C::new((a - energy) / norm, 0.0),
    ];
    let overlap = get(base, 0).conj() * raw[0] + get(base, 1).conj() * raw[1];
    let phase = overlap.conj() / overlap.re.hypot(overlap.im);
    let out = [raw[0] * phase, raw[1] * phase];
    [out[0].re, out[0].im, out[1].re, out[1].im]
}

#[test]
fn every_complex_eigenvector_gradient_and_hessian_entry_has_four_step_fits() {
    let config = SpinorTdaConfig::default();
    let paths = [
        spinor_tda_eigenpair_taylor(&jets(1.0, 0.0), 0, &config).unwrap(),
        spinor_tda_eigenpair_taylor(&jets(0.0, 1.0), 0, &config).unwrap(),
        spinor_tda_eigenpair_taylor(&jets(1.0, 1.0), 0, &config).unwrap(),
    ];
    for path in &paths {
        assert!(path.equation_residuals.iter().all(|r| *r < 1e-12));
        assert!(path.normalization_defects.iter().all(|r| *r < 1e-12));
        assert!(path.nearest_gap > 1.0);
    }
    let base = &paths[0].vectors[0];
    let gx = flatten(&paths[0].vectors[1]);
    let gy = flatten(&paths[1].vectors[1]);
    let xx = flatten(&paths[0].vectors[2]);
    let yy = flatten(&paths[1].vectors[2]);
    let sum = flatten(&paths[2].vectors[2]);
    let mut analytic = Vec::new();
    let mut names = Vec::new();
    for axis in 0..2 {
        for component in 0..4 {
            names.push(format!("eigenvector gradient {axis},component{component}"));
            analytic.push(if axis == 0 {
                gx[component]
            } else {
                gy[component]
            });
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            for c in 0..4 {
                names.push(format!("eigenvector Hessian {i},{j},component{c}"));
                analytic.push(if i == j {
                    2.0 * if i == 0 { xx[c] } else { yy[c] }
                } else {
                    sum[c] - xx[c] - yy[c]
                });
            }
        }
    }
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[
            [0.08, 0.04, 0.02, 0.01],
            [0.16, 0.08, 0.04, 0.02],
            [0.04, 0.02, 0.01, 0.005],
        ],
        |steps| {
            let mut errors = vec![[0.0; 4]; 24];
            let at = |x: f64, y: f64| exact(0.17 + x, -0.23 + y, base);
            let zero = at(0.0, 0.0);
            for (step, &h) in steps.iter().enumerate() {
                for axis in 0..2 {
                    let p = at(
                        if axis == 0 { h } else { 0.0 },
                        if axis == 1 { h } else { 0.0 },
                    );
                    let m = at(
                        if axis == 0 { -h } else { 0.0 },
                        if axis == 1 { -h } else { 0.0 },
                    );
                    for c in 0..4 {
                        errors[4 * axis + c][step] =
                            ((p[c] - m[c]) / (2.0 * h) - analytic[4 * axis + c]).abs();
                    }
                }
                for i in 0..2 {
                    for j in 0..2 {
                        let derivative = if i == j {
                            let p = at(if i == 0 { h } else { 0.0 }, if i == 1 { h } else { 0.0 });
                            let m =
                                at(if i == 0 { -h } else { 0.0 }, if i == 1 { -h } else { 0.0 });
                            std::array::from_fn::<_, 4, _>(|c| {
                                (p[c] - 2.0 * zero[c] + m[c]) / (h * h)
                            })
                        } else {
                            let pp = at(h, h);
                            let pm = at(h, -h);
                            let mp = at(-h, h);
                            let mm = at(-h, -h);
                            std::array::from_fn(|c| (pp[c] - pm[c] - mp[c] + mm[c]) / (4.0 * h * h))
                        };
                        for c in 0..4 {
                            let k = 8 + 4 * (2 * i + j) + c;
                            errors[k][step] = (derivative[c] - analytic[k]).abs();
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all complex eigenvector derivatives", &failures);
}

#[test]
fn eigenvector_jets_preserve_complex_basis_phase_covariance_and_gap_diagnostics() {
    let config = SpinorTdaConfig::default();
    let operator = jets(0.7, -0.4);
    let original = spinor_tda_eigenpair_taylor(&operator, 0, &config).unwrap();
    let phases = [
        C::new(0.37_f64.cos(), 0.37_f64.sin()),
        C::new((-0.61_f64).cos(), (-0.61_f64).sin()),
    ];
    let rotated = operator
        .iter()
        .map(|a| {
            let mut b = BlochMatrix::zeros(2, 2);
            for i in 0..2 {
                for j in 0..2 {
                    let v = phases[i].conj() * C::new(a.re[(i, j)], a.im[(i, j)]) * phases[j];
                    b.re[(i, j)] = v.re;
                    b.im[(i, j)] = v.im;
                }
            }
            b
        })
        .collect::<Vec<_>>();
    let actual = spinor_tda_eigenpair_taylor(&rotated, 0, &config).unwrap();
    let overlap = (0..2)
        .map(|i| {
            get(&actual.vectors[0], i).conj() * phases[i].conj() * get(&original.vectors[0], i)
        })
        .fold(C::new(0.0, 0.0), |a, b| a + b);
    let norm = overlap.re.hypot(overlap.im);
    assert!((norm - 1.0).abs() < 1e-12);
    let phase = overlap.conj() / norm;
    for k in 0..3 {
        assert!((actual.energies[k] - original.energies[k]).abs() < 1e-12);
        for i in 0..2 {
            let delta = get(&actual.vectors[k], i)
                - phases[i].conj() * get(&original.vectors[k], i) * phase;
            assert!(delta.re.hypot(delta.im) < 1e-12);
        }
    }
    let degenerate = vec![BlochMatrix::zeros(2, 2); 3];
    assert!(matches!(
        spinor_tda_eigenpair_taylor(&degenerate, 0, &config),
        Err(dft_rs::DftError::Unsupported { .. })
    ));
    let single = vec![
        {
            let mut a = BlochMatrix::zeros(1, 1);
            a.re[(0, 0)] = 0.5;
            a
        },
        BlochMatrix::zeros(1, 1),
    ];
    let result = spinor_tda_eigenpair_taylor(&single, 0, &config).unwrap();
    assert!(result.nearest_gap.is_infinite());
    assert!(result.vectors[1]
        .re
        .as_slice()
        .iter()
        .chain(result.vectors[1].im.as_slice())
        .all(|v| *v == 0.0));
    let overflow = vec![
        matrix([0.5, 1.0, 0.0, 0.0]),
        matrix([0.0, 0.0, f64::MAX, 0.0]),
        BlochMatrix::zeros(2, 2),
    ];
    let failure = match spinor_tda_eigenpair_taylor(&overflow, 0, &config) {
        Ok(_) => panic!("overflow must be refused"),
        Err(error) => error,
    };
    println!("overflow diagnostic {failure:?}");
    assert!(
        matches!(failure,dft_rs::DftError::ResponseNotConverged{residual,..} if residual.is_infinite())
    );
}

#[test]
fn all_roots_match_known_unitary_path_energy_and_vector_coefficients() {
    use dft_rs::dfpt::orbitals::{adjoint, matmul};
    let config = SpinorTdaConfig::default();
    let mut energies_checked = 0;
    for n in [2, 3, 5, 7, 15] {
        let mut q = BlochMatrix::zeros(n, n);
        let mut k = BlochMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let angle = 2.0 * std::f64::consts::PI * (i * j) as f64 / n as f64;
                q.re[(i, j)] = angle.cos() / (n as f64).sqrt();
                q.im[(i, j)] = angle.sin() / (n as f64).sqrt();
                if i > j {
                    let re = 0.013 * ((i + 3 * j + 1) as f64).sin();
                    let im = 0.009 * ((2 * i + j + 2) as f64).cos();
                    k.re[(i, j)] = re;
                    k.re[(j, i)] = -re;
                    k.im[(i, j)] = im;
                    k.im[(j, i)] = im;
                }
            }
        }
        let diagonal = (0..3)
            .map(|order| {
                let mut a = BlochMatrix::zeros(n, n);
                for i in 0..n {
                    a.re[(i, i)] = match order {
                        0 => 0.5 + 0.9 * i as f64,
                        1 => 0.04 * ((i + 1) as f64).sin(),
                        _ => -0.02 * ((i + 2) as f64).cos(),
                    };
                }
                a
            })
            .collect::<Vec<_>>();
        let combine = |a: &BlochMatrix, b: &BlochMatrix, scale: f64| {
            let mut c = a.clone();
            for i in 0..n {
                for j in 0..n {
                    c.re[(i, j)] += scale * b.re[(i, j)];
                    c.im[(i, j)] += scale * b.im[(i, j)];
                }
            }
            c
        };
        let commutator =
            |a: &BlochMatrix| combine(&matmul(&k, a).unwrap(), &matmul(a, &k).unwrap(), -1.0);
        let first_comm = commutator(&diagonal[0]);
        let second_comm = commutator(&first_comm);
        let raw = [
            diagonal[0].clone(),
            combine(&diagonal[1], &first_comm, 1.0),
            combine(
                &combine(&diagonal[2], &commutator(&diagonal[1]), 1.0),
                &second_comm,
                0.5,
            ),
        ];
        let qt = adjoint(&q).unwrap();
        let operator = raw
            .iter()
            .map(|a| matmul(&matmul(&q, a).unwrap(), &qt).unwrap())
            .collect::<Vec<_>>();
        let first = matmul(&q, &k).unwrap();
        let second = matmul(&first, &k).unwrap();
        for root in 0..n {
            let actual = spinor_tda_eigenpair_taylor(&operator, root, &config).unwrap();
            let legacy = spinor_tda_pole_taylor(&operator, root, &config).unwrap();
            let overlap = (0..n)
                .map(|i| {
                    get(&actual.vectors[0], i).conj() * C::new(q.re[(i, root)], q.im[(i, root)])
                })
                .fold(C::new(0.0, 0.0), |a, b| a + b);
            let phase = overlap.conj() / overlap.re.hypot(overlap.im);
            for order in 0..3 {
                assert!((actual.energies[order] - diagonal[order].re[(root, root)]).abs() < 1e-11);
                assert!((legacy[order] - diagonal[order].re[(root, root)]).abs() < 1e-11);
                energies_checked += 1;
                for i in 0..n {
                    let matrix = match order {
                        0 => &q,
                        1 => &first,
                        _ => &second,
                    };
                    let scale = if order == 2 { 0.5 } else { 1.0 };
                    let expected =
                        C::new(scale * matrix.re[(i, root)], scale * matrix.im[(i, root)]) * phase;
                    let error = get(&actual.vectors[order], i) - expected;
                    assert!(error.re.hypot(error.im) < 1e-11);
                }
            }
        }
    }
    assert_eq!(energies_checked, 96);
}
