// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full-spin excitation-space contracts with independently assembled HF
//! spin couplings; no nuclear derivative is inferred from these spectra.

use dft_rs::linalg::Matrix;
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::spinor_tda::spinor_tda_pole_taylor;
use dft_rs::response::spinor_tda::{spinor_tda_states_with_kernel, SpinorTdaConfig};
use dft_rs::{DftError, Result};

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

fn identity(n: usize) -> BlochMatrix {
    BlochMatrix {
        re: Matrix::identity(n),
        im: Matrix::zeros(n, n),
    }
}

fn orbitals(phases: [f64; 4]) -> BlochMatrix {
    let mut result = BlochMatrix::zeros(4, 4);
    for (column, row) in [0, 2, 1, 3].into_iter().enumerate() {
        result.re[(row, column)] = phases[column].cos();
        result.im[(row, column)] = phases[column].sin();
    }
    result
}

fn integral(i: usize, j: usize, k: usize, l: usize) -> f64 {
    if i != j && k != l {
        0.13
    } else if i == j && k == l && i != k {
        0.29
    } else if i == j && j == k && k == l {
        if i == 0 {
            0.7
        } else {
            0.6
        }
    } else {
        0.0
    }
}

fn hf_kernel(density: &BlochMatrix) -> Result<BlochMatrix> {
    let mut result = BlochMatrix::zeros(4, 4);
    for s in 0..2 {
        for t in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            let exchange = integral(i, k, j, l);
                            result.re[(2 * s + i, 2 * t + j)] -=
                                exchange * density.re[(2 * s + k, 2 * t + l)];
                            result.im[(2 * s + i, 2 * t + j)] -=
                                exchange * density.im[(2 * s + k, 2 * t + l)];
                            if s == t {
                                for spin in 0..2 {
                                    let coulomb = integral(i, j, k, l);
                                    result.re[(2 * s + i, 2 * s + j)] +=
                                        coulomb * density.re[(2 * spin + k, 2 * spin + l)];
                                    result.im[(2 * s + i, 2 * s + j)] +=
                                        coulomb * density.im[(2 * spin + k, 2 * spin + l)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(result)
}

#[test]
fn complete_spinor_space_recovers_all_triplet_components_and_singlet() {
    for phases in [[0.0; 4], [0.27, -0.43, 0.19, 0.61]] {
        let states = spinor_tda_states_with_kernel(
            &orbitals(phases),
            &[-0.7, -0.7, 0.4, 0.4],
            2,
            &identity(4),
            &hf_kernel,
            &SpinorTdaConfig {
                states: 4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(states.len(), 4);
        for (state, expected) in states.iter().zip([0.81, 0.81, 0.81, 1.07]) {
            assert!((state.energy - expected).abs() < 1e-12);
            assert!(state.residual < 1e-12);
            let norm = state
                .amplitudes
                .re
                .as_slice()
                .iter()
                .zip(state.amplitudes.im.as_slice())
                .map(|(r, i)| r * r + i * i)
                .sum::<f64>();
            assert!((norm - 1.0).abs() < 1e-12);
            let trace = (0..4)
                .map(|i| {
                    state.transition_density.re[(i, i)].abs()
                        + state.transition_density.im[(i, i)].abs()
                })
                .sum::<f64>();
            assert!(trace < 1e-12);
        }
        for state in &states[..3] {
            assert!(state.nearest_gap < 1e-12);
        }
        assert!((states[3].nearest_gap - 0.26).abs() < 1e-12);
        if phases[0] != 0.0 {
            assert!(states.iter().any(|state| state
                .amplitudes
                .im
                .as_slice()
                .iter()
                .any(|x| x.abs() > 0.05)));
        }
    }
}

#[test]
fn transition_kernel_must_be_complex_linear_and_budgeted() {
    let invalid = |density: &BlochMatrix| -> Result<BlochMatrix> {
        let mut out = density.clone();
        for i in 0..4 {
            for j in 0..4 {
                out.re[(i, j)] += density.re[(j, i)];
                out.im[(i, j)] -= density.im[(j, i)];
            }
        }
        Ok(out)
    };
    let result = spinor_tda_states_with_kernel(
        &orbitals([0.0; 4]),
        &[-0.7, -0.7, 0.4, 0.4],
        2,
        &identity(4),
        &invalid,
        &SpinorTdaConfig::default(),
    );
    assert!(matches!(result, Err(DftError::InvalidInput(_))));
    let result = spinor_tda_states_with_kernel(
        &orbitals([0.0; 4]),
        &[-0.7, -0.7, 0.4, 0.4],
        2,
        &identity(4),
        &hf_kernel,
        &SpinorTdaConfig {
            memory_mb: 0,
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(DftError::ResourceLimit { .. })));
}

#[test]
fn full_spinor_spectrum_is_invariant_under_global_complex_spin_rotation() {
    use dft_rs::response::complex::Complex as C;
    let base = orbitals([0.27, -0.43, 0.19, 0.61]);
    let mut rotated = BlochMatrix::zeros(4, 4);
    let theta = 0.61_f64;
    let phase = 0.43_f64;
    let c = C::new(theta.cos(), 0.0);
    let p = C::new(phase.cos() * theta.sin(), phase.sin() * theta.sin());
    for spatial in 0..2 {
        for column in 0..4 {
            let up = C::new(base.re[(spatial, column)], base.im[(spatial, column)]);
            let down = C::new(
                base.re[(2 + spatial, column)],
                base.im[(2 + spatial, column)],
            );
            let first = c * up - p.conj() * down;
            let second = p * up + c * down;
            rotated.re[(spatial, column)] = first.re;
            rotated.im[(spatial, column)] = first.im;
            rotated.re[(2 + spatial, column)] = second.re;
            rotated.im[(2 + spatial, column)] = second.im;
        }
    }
    let states = spinor_tda_states_with_kernel(
        &rotated,
        &[-0.7, -0.7, 0.4, 0.4],
        2,
        &identity(4),
        &hf_kernel,
        &SpinorTdaConfig {
            states: 4,
            ..Default::default()
        },
    )
    .unwrap();
    for (state, expected) in states.iter().zip([0.81, 0.81, 0.81, 1.07]) {
        assert!((state.energy - expected).abs() < 1e-12);
    }
}

fn two_by_two(values: [f64; 4]) -> BlochMatrix {
    let [a, d, re, im] = values;
    let mut out = BlochMatrix::zeros(2, 2);
    out.re[(0, 0)] = a;
    out.re[(1, 1)] = d;
    out.re[(0, 1)] = re;
    out.re[(1, 0)] = re;
    out.im[(0, 1)] = im;
    out.im[(1, 0)] = -im;
    out
}

#[test]
fn complex_pole_has_all_two_parameter_gradient_and_hessian_fits() {
    let coefficients = [
        [0.7, 1.9, 0.21, 0.17],
        [0.31, -0.23, 0.09, -0.07],
        [-0.19, 0.11, -0.06, 0.08],
        [0.07, -0.03, 0.02, 0.04],
        [0.05, 0.04, -0.03, 0.02],
        [-0.04, 0.09, 0.01, -0.02],
    ];
    let path = |u: f64, v: f64| {
        let first = std::array::from_fn(|k| u * coefficients[1][k] + v * coefficients[2][k]);
        let second = std::array::from_fn(|k| {
            u * u * coefficients[3][k] + u * v * coefficients[4][k] + v * v * coefficients[5][k]
        });
        spinor_tda_pole_taylor(
            &[
                two_by_two(coefficients[0]),
                two_by_two(first),
                two_by_two(second),
            ],
            0,
            &SpinorTdaConfig::default(),
        )
        .unwrap()
    };
    let pu = path(1.0, 0.0);
    let pv = path(0.0, 1.0);
    let puv = path(1.0, 1.0);
    let mixed = puv[2] - pu[2] - pv[2];
    let analytic = [pu[1], pv[1], 2.0 * pu[2], mixed, mixed, 2.0 * pv[2]];
    // Closed-form lower eigenvalue of a complex Hermitian 2x2 matrix,
    // independently evaluated without the production spectral recurrence.
    let energy = |u: f64, v: f64| {
        let a = 0.7 + 0.31 * u - 0.19 * v + 0.07 * u * u + 0.05 * u * v - 0.04 * v * v;
        let d = 1.9 - 0.23 * u + 0.11 * v - 0.03 * u * u + 0.04 * u * v + 0.09 * v * v;
        let re = 0.21 + 0.09 * u - 0.06 * v + 0.02 * u * u - 0.03 * u * v + 0.01 * v * v;
        let im = 0.17 - 0.07 * u + 0.08 * v + 0.04 * u * u + 0.02 * u * v - 0.02 * v * v;
        0.5 * (a + d - ((a - d).powi(2) + 4.0 * (re * re + im * im)).sqrt())
    };
    let names = [
        "gradient u",
        "gradient v",
        "Hessian uu",
        "Hessian uv",
        "Hessian vu",
        "Hessian vv",
    ]
    .map(String::from);
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.16, 0.08, 0.04, 0.02], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 6];
            for (k, &h) in steps.iter().enumerate() {
                let at = |s: f64| {
                    let g0 = stencil.first(
                        energy(s, 0.0),
                        energy(-s, 0.0),
                        energy(2.0 * s, 0.0),
                        energy(-2.0 * s, 0.0),
                        s,
                    );
                    let g1 = stencil.first(
                        energy(0.0, s),
                        energy(0.0, -s),
                        energy(0.0, 2.0 * s),
                        energy(0.0, -2.0 * s),
                        s,
                    );
                    let uu = (energy(s, 0.0) - 2.0 * energy(0.0, 0.0) + energy(-s, 0.0)) / (s * s);
                    let vv = (energy(0.0, s) - 2.0 * energy(0.0, 0.0) + energy(0.0, -s)) / (s * s);
                    let uv = (energy(s, s) - energy(s, -s) - energy(-s, s) + energy(-s, -s))
                        / (4.0 * s * s);
                    [g0, g1, uu, uv, uv, vv]
                };
                let mut estimated = at(h);
                if stencil == stencils::Stencil::Fourth {
                    let coarse = at(2.0 * h);
                    for i in 2..6 {
                        estimated[i] = (4.0 * estimated[i] - coarse[i]) / 3.0;
                    }
                }
                for i in 0..6 {
                    errors[i][k] = (estimated[i] - analytic[i]).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("complex isolated pole", &failures);
    let degenerate = two_by_two([1.0, 1.0, 0.0, 0.0]);
    assert!(matches!(
        spinor_tda_pole_taylor(
            &[degenerate.clone(), degenerate],
            0,
            &SpinorTdaConfig::default()
        ),
        Err(DftError::Unsupported { .. })
    ));
}
