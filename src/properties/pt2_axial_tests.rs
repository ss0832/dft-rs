// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit determinants of rotated Slater overlaps, independent of the
//! sparse derivative-state action. Every nuclear/field component is fitted.
use super::*;
use crate::acceptance::components::{assert_no_failures, fit_components};

fn amplitude(n: [usize; 2], o: [usize; 2], phase: f64, imaginary: bool) -> [Vec<Complex>; 3] {
    std::array::from_fn(|c| {
        let (s, u) = [(0, 0), (0, 1), (1, 1)][c];
        let (ns, nu, vs, vu) = (o[s], o[u], n[s] - o[s], n[u] - o[u]);
        let value = |i: usize, a: usize, j: usize, b: usize| {
            let x = (1 + i + 3 * a + 7 * j + 11 * b + 13 * c) as f64;
            0.047 * (0.29 * x + phase).sin() + 0.021 * (0.17 * x - phase).cos()
        };
        let mut result = vec![Complex::ZERO; ns * vs * nu * vu];
        for i in 0..ns {
            for a in 0..vs {
                for j in 0..nu {
                    for b in 0..vu {
                        let v = if s == u {
                            if i == j || a == b {
                                0.0
                            } else {
                                let sign = if (i < j) == (a < b) { 1.0 } else { -1.0 };
                                sign * value(i.min(j), a.min(b), i.max(j), a.max(b))
                            }
                        } else {
                            value(i, a, j, b)
                        };
                        result[((i * vs + a) * nu + j) * vu + b] = if imaginary {
                            Complex::new(0.0, v)
                        } else {
                            Complex::from(v)
                        };
                    }
                }
            }
        }
        result
    })
}

fn generator(n: [usize; 2], axis: usize, imaginary: bool) -> [ComplexMatrix; 2] {
    std::array::from_fn(|s| {
        let mut result = ComplexMatrix::zeros(n[s], n[s]);
        for i in 0..n[s] {
            for j in i..n[s] {
                let v = 0.31 * ((2 + i + 3 * j + 5 * s + 7 * axis) as f64 * 0.23).sin()
                    + 0.12 * ((1 + 2 * i + j + 3 * s + 2 * axis) as f64 * 0.19).cos();
                if imaginary {
                    result.im[(i, j)] = v;
                    result.im[(j, i)] = v;
                } else if i != j {
                    result.re[(i, j)] = v;
                    result.re[(j, i)] = -v;
                }
            }
        }
        result
    })
}

fn exponential(k: &ComplexMatrix, t: f64) -> ComplexMatrix {
    let n = k.re.rows;
    let mut result = ComplexMatrix::zeros(n, n);
    for i in 0..n {
        result.re[(i, i)] = 1.0;
    }
    let mut term = result.clone();
    for order in 1..=32 {
        // Plain dense matrix multiplication here is independent of the
        // determinant-connection action used by the production provider.
        let mut next = ComplexMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    let a = Complex::new(term.re[(i, l)], term.im[(i, l)]);
                    let b = Complex::new(k.re[(l, j)], k.im[(l, j)]);
                    let z = a * b * (t / order as f64);
                    next.re[(i, j)] += z.re;
                    next.im[(i, j)] += z.im;
                }
            }
        }
        term = next;
        for i in 0..n {
            for j in 0..n {
                result.re[(i, j)] += term.re[(i, j)];
                result.im[(i, j)] += term.im[(i, j)];
            }
        }
    }
    result
}

#[path = "../../tests/support/slater.rs"]
mod slater;
use slater::{finite_state, slater_overlap};

#[test]
fn every_pair_axial_component_matches_independent_normalized_slater_overlaps() {
    let n = [4, 5];
    let o = [2, 2];
    let base = amplitude(n, o, 0.13, false);
    let real = base.each_ref().map(|v| v.iter().map(|z| z.re).collect());
    let state = PairAxialState::new(n, o, &real).unwrap();
    let weights = Pt2Coefficients {
        same_spin: 0.31,
        opposite_spin: 0.47,
    };
    let mut failures = Vec::new();
    for x in 0..6 {
        let kr = generator(n, x, false);
        let ar = amplitude(n, o, 0.37 + 0.29 * x as f64, false);
        for b in 0..3 {
            let kb = generator(n, 3 + b, true);
            let ab = amplitude(n, o, -0.43 - 0.19 * b as f64, true);
            let analytic = state
                .overlap(
                    &PairStateTangent {
                        connection: &kr,
                        amplitudes: &ar,
                    },
                    &PairStateTangent {
                        connection: &kb,
                        amplitudes: &ab,
                    },
                    weights,
                    1e-12,
                )
                .unwrap();
            assert!(analytic.cancellation_defect < 1e-13);
            assert!(analytic.connection_defect < 1e-13);
            assert!(analytic
                .channel_norms
                .iter()
                .all(|x| x.is_finite() && *x > 0.0));
            assert!(analytic.norm > 1.0);
            assert!(
                analytic.norm_derivative.is_finite() && analytic.magnetic_connection.is_finite()
            );
            let zero = state
                .overlap(
                    &PairStateTangent {
                        connection: &kr,
                        amplitudes: &ar,
                    },
                    &PairStateTangent {
                        connection: &kb,
                        amplitudes: &ab,
                    },
                    Pt2Coefficients {
                        same_spin: 0.0,
                        opposite_spin: 0.0,
                    },
                    1e-12,
                )
                .unwrap();
            assert_eq!(zero.value, zero.parent);
            let targets = [
                analytic.value,
                analytic.parent,
                analytic.channel_bilinears[0],
                analytic.channel_bilinears[1],
                analytic.channel_bilinears[2],
            ];
            let names: Vec<_> = (0..5)
                .map(|c| format!("MP1 axial nuclear={x} B={b} channel={c}"))
                .collect();
            fit_components(
                &names,
                &[
                    [0.08, 0.06, 0.04, 0.03],
                    [0.02, 0.015, 0.01, 0.0075],
                    [0.16, 0.12, 0.08, 0.06],
                ],
                |steps| {
                    let mut errors = vec![[0.0; 4]; 5];
                    for (k, &h) in steps.iter().enumerate() {
                        let mut mixed = [0.0; 5];
                        for sr in [-1.0, 1.0] {
                            for sb in [-1.0, 1.0] {
                                let ur: Vec<_> =
                                    kr.iter().map(|m| exponential(m, sr * h)).collect();
                                let ub: Vec<_> =
                                    kb.iter().map(|m| exponential(m, sb * h)).collect();
                                let mut cross = vec![vec![Complex::ZERO; n[0] + n[1]]; n[0] + n[1]];
                                for s in 0..2 {
                                    let offset = if s == 0 { 0 } else { n[0] };
                                    for i in 0..n[s] {
                                        for j in 0..n[s] {
                                            for p in 0..n[s] {
                                                cross[offset + i][offset + j] += Complex::new(
                                                    ur[s].re[(p, i)],
                                                    -ur[s].im[(p, i)],
                                                ) * Complex::new(
                                                    ub[s].re[(p, j)],
                                                    ub[s].im[(p, j)],
                                                );
                                            }
                                        }
                                    }
                                }
                                for c in 0..5 {
                                    let channel = if c >= 2 { Some(c - 2) } else { None };
                                    let w = if c == 1 {
                                        Pt2Coefficients {
                                            same_spin: 0.0,
                                            opposite_spin: 0.0,
                                        }
                                    } else {
                                        weights
                                    };
                                    let left = finite_state(n, o, &base, &ar, sr * h, channel, w);
                                    let right = finite_state(n, o, &base, &ab, sb * h, channel, w);
                                    let scale = if c == 0 {
                                        let nl = left
                                            .iter()
                                            .map(|(_, z)| z.re * z.re + z.im * z.im)
                                            .sum::<f64>();
                                        let nr = right
                                            .iter()
                                            .map(|(_, z)| z.re * z.re + z.im * z.im)
                                            .sum::<f64>();
                                        (nl * nr).sqrt()
                                    } else {
                                        1.0
                                    };
                                    mixed[c] +=
                                        sr * sb * slater_overlap(&left, &right, &cross).im / scale;
                                }
                            }
                        }
                        for c in 0..5 {
                            errors[c][k] = (mixed[c] / (4.0 * h * h) - targets[c]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    assert_no_failures(
        "all normalized pair-wavefunction axial components",
        &failures,
    );
}

#[test]
fn pair_states_have_no_fixed_word_orbital_limit_and_recover_one_electron_response() {
    let n = 130;
    let amplitudes = std::array::from_fn(|_| Vec::new());
    // Same-spin [i,a,j,b] blocks include the identically vanishing diagonal
    // pairs even for a one-electron determinant.
    let mut amplitudes: [Vec<f64>; 3] = amplitudes;
    amplitudes[0] = vec![0.0; (n - 1) * (n - 1)];
    let state = PairAxialState::new([n, 0], [1, 0], &amplitudes).unwrap();
    let mut r = [ComplexMatrix::zeros(n, n), ComplexMatrix::zeros(0, 0)];
    let mut b = r.clone();
    r[0].re[(129, 0)] = 0.2;
    r[0].re[(0, 129)] = -0.2;
    b[0].im[(129, 0)] = 0.3;
    b[0].im[(0, 129)] = 0.3;
    let zeros = amplitudes.map(|v| v.into_iter().map(Complex::from).collect());
    let result = state
        .overlap(
            &PairStateTangent {
                connection: &r,
                amplitudes: &zeros,
            },
            &PairStateTangent {
                connection: &b,
                amplitudes: &zeros,
            },
            Pt2Coefficients {
                same_spin: 0.31,
                opposite_spin: 0.47,
            },
            1e-12,
        )
        .unwrap();
    assert!((result.value - 0.06).abs() < 1e-15);
    assert_eq!(result.value, result.parent);
    assert_eq!(result.norm, 1.0);
}

#[test]
fn amplitude_berry_connection_matches_a_normalized_two_state_model_and_global_phase() {
    let c = 0.47_f64;
    let (t, tr, tb) = (0.2_f64, 0.3_f64, 0.4_f64);
    let state = PairAxialState::new([2, 2], [1, 1], &[vec![0.0], vec![t], vec![0.0]]).unwrap();
    let r = [ComplexMatrix::zeros(2, 2), ComplexMatrix::zeros(2, 2)];
    let mut b = r.clone();
    let ar = [
        vec![Complex::ZERO],
        vec![Complex::from(tr)],
        vec![Complex::ZERO],
    ];
    let ab = [
        vec![Complex::ZERO],
        vec![Complex::new(0.0, tb)],
        vec![Complex::ZERO],
    ];
    let weights = Pt2Coefficients {
        same_spin: 0.0,
        opposite_spin: c,
    };
    let result = state
        .overlap(
            &PairStateTangent {
                connection: &r,
                amplitudes: &ar,
            },
            &PairStateTangent {
                connection: &b,
                amplitudes: &ab,
            },
            weights,
            1e-12,
        )
        .unwrap();
    // Direct derivatives of (1,sqrt(c)*t)/sqrt(1+c*t^2).
    let exact = c * tr * tb / (1.0 + c * t * t).powi(2);
    assert!((result.value - exact).abs() < 1e-15);
    assert!((result.magnetic_connection - c * t * tb).abs() < 1e-15);
    let mut failures = Vec::new();
    fit_components(
        &["normalized two-state amplitude Berry term".into()],
        &[[0.08, 0.06, 0.04, 0.03]],
        |steps| {
            let errors = steps.map(|h| {
                let mut difference = 0.0;
                for sr in [-1.0, 1.0] {
                    for sb in [-1.0, 1.0] {
                        let a = t + sr * h * tr;
                        let norm =
                            ((1.0 + c * a * a) * (1.0 + c * (t * t + h * h * tb * tb))).sqrt();
                        difference += sr * sb * c * a * (sb * h * tb) / norm;
                    }
                }
                (difference / (4.0 * h * h) - result.value).abs()
            });
            vec![errors]
        },
        &mut failures,
    );
    assert_no_failures("amplitude Berry connection", &failures);
    // A field-dependent global orbital phase changes the Berry connection
    // but cannot change this normalized mixed derivative overlap.
    for spin in &mut b {
        for i in 0..2 {
            spin.im[(i, i)] = 0.37;
        }
    }
    let shifted = state
        .overlap(
            &PairStateTangent {
                connection: &r,
                amplitudes: &ar,
            },
            &PairStateTangent {
                connection: &b,
                amplitudes: &ab,
            },
            weights,
            1e-12,
        )
        .unwrap();
    assert!((shifted.value - result.value).abs() < 1e-15);
    assert!(
        (shifted.magnetic_connection - result.magnetic_connection - 0.74 * result.norm).abs()
            < 1e-15
    );
}

#[test]
fn signed_pair_norm_and_connection_failures_are_explicit() {
    let n = [4, 5];
    let o = [2, 2];
    let base = amplitude(n, o, 0.13, false);
    let real = base.each_ref().map(|v| v.iter().map(|z| z.re).collect());
    assert!(matches!(
        PairAxialState::with_memory_limit(n, o, &real, 0),
        Err(DftError::ResourceLimit {
            operation: "pair-state derivative overlaps",
            ..
        })
    ));
    let state = PairAxialState::new(n, o, &real).unwrap();
    let r = generator(n, 0, false);
    let b = generator(n, 0, true);
    let ar = amplitude(n, o, 0.31, false);
    let ab = amplitude(n, o, 0.21, true);
    let result = state.overlap(
        &PairStateTangent {
            connection: &r,
            amplitudes: &ar,
        },
        &PairStateTangent {
            connection: &b,
            amplitudes: &ab,
        },
        Pt2Coefficients {
            same_spin: -1e9,
            opposite_spin: -1e9,
        },
        1e-12,
    );
    assert!(matches!(result, Err(DftError::InvalidInput(_))));
    let mut bad = r.clone();
    bad[0].re[(0, 0)] = 0.01;
    let result = state.overlap(
        &PairStateTangent {
            connection: &bad,
            amplitudes: &ar,
        },
        &PairStateTangent {
            connection: &b,
            amplitudes: &ab,
        },
        Pt2Coefficients {
            same_spin: 0.31,
            opposite_spin: 0.47,
        },
        1e-12,
    );
    assert!(matches!(result, Err(DftError::ResponseNotConverged { .. })));
}
