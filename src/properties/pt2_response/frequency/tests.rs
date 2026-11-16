// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::acceptance::{components, stencils::Stencil};
use crate::linalg::{solve_linear, Matrix};
use crate::periodic::BlochMatrix;
use crate::properties::pt2_response::taylor_response;

const MODES: usize = 3;
const COEFFICIENTS: Pt2Coefficients = Pt2Coefficients {
    same_spin: 0.31,
    opposite_spin: 0.57,
};

fn decoded(p: usize) -> [usize; 4] {
    [p >> 3, (p >> 2) & 1, (p >> 1) & 1, p & 1]
}

fn packed([i, a, j, b]: [usize; 4]) -> usize {
    8 * i + 4 * a + 2 * j + b
}

fn raw_block(channel: usize, amplitude: Complex, slope: Complex) -> Vec<Complex> {
    (0..16)
        .map(|p| {
            if channel == 1 {
                amplitude + slope * ((2 * p + 1) as f64).sin()
            } else {
                let [i, a, j, b] = decoded(p);
                amplitude * (0.5 * (j as f64 - i as f64) * (b as f64 - a as f64))
            }
        })
        .collect()
}

fn problem() -> Pt2Problem<f64> {
    Pt2Problem {
        energies: [
            vec![-1.19, -0.71, 0.26, 0.82],
            vec![-1.03, -0.62, 0.34, 1.08],
        ],
        occupied: [2, 2],
        ovov: std::array::from_fn(|c| {
            raw_block(
                c,
                Complex::from(0.22 + 0.03 * c as f64),
                Complex::from(0.035),
            )
            .into_iter()
            .map(|v| v.re)
            .collect()
        }),
    }
}

fn matrix_element(f: &ComplexMatrix, i: usize, j: usize) -> Complex {
    Complex::new(f.re[(i, j)], f.im[(i, j)])
}

fn fixture_vertex(first: usize, second: usize, z: Complex, mixed: bool) -> Pt2FrequencyVertex {
    let a = first as f64 + 1.0;
    let b = second as f64 + 1.0;
    let scale = if mixed { a + b + 0.2 * a * b } else { a };
    let block = |sign: f64| {
        std::array::from_fn(|c| {
            let base = 0.024 * scale + 0.013 * c as f64;
            let response = if mixed {
                z * z * (0.017 * scale)
            } else {
                z * (sign * 0.061 * scale)
            };
            raw_block(
                c,
                Complex::from(base) + response,
                Complex::from(0.029 * scale) - response * 0.32,
            )
        })
    };
    let orbital_generator = std::array::from_fn(|s| {
        let mut generator = ComplexMatrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                let even = 0.031 * (a + (i + j + 1) as f64).cos() + 0.009 * b * (s + 1) as f64;
                let odd = 0.036 * scale * (i as f64 - j as f64);
                let v = if mixed {
                    Complex::from(even) + z * z * (0.017 * even)
                } else {
                    Complex::from(even) + z * odd
                };
                generator.re[(i, j)] = v.re;
                generator.im[(i, j)] = v.im;
            }
        }
        generator
    });
    Pt2FrequencyVertex {
        coupling_left: block(-1.0),
        coupling_right: block(1.0),
        orbital_generator,
    }
}

fn vertices(a: usize, b: usize, z: Complex) -> [Pt2FrequencyVertex; 3] {
    [
        fixture_vertex(a, a, z, false),
        fixture_vertex(b, b, -z, false),
        fixture_vertex(a.min(b), a.max(b), z, true),
    ]
}

fn unique_value(raw: &[Complex], c: usize, state: [usize; 4]) -> Complex {
    let [i, a, j, b] = state;
    let direct = raw[packed(state)];
    if c == 1 {
        direct
    } else {
        direct - raw[packed([i, b, j, a])]
    }
}

// The oracle uses unique determinants. Same-spin channels have one
// determinant in this space, while opposite spin has all 16 products.
fn unique_action(c: usize, f: &[ComplexMatrix; 2], row: [usize; 4], col: [usize; 4]) -> Complex {
    let [i, a, j, b] = row;
    let [k, u, l, v] = col;
    let (s, t) = SPINS[c];
    let mut value = Complex::ZERO;
    if a == u && j == l && b == v {
        value += matrix_element(&f[s], k, i);
    }
    if i == k && a == u && b == v {
        value += matrix_element(&f[t], l, j);
    }
    if i == k && j == l && b == v {
        value -= matrix_element(&f[s], 2 + a, 2 + u);
    }
    if i == k && a == u && j == l {
        value -= matrix_element(&f[t], 2 + b, 2 + v);
    }
    value
}

fn dense_floquet_energy(
    problem: &Pt2Problem<f64>,
    vertices: &[Pt2FrequencyVertex; 3],
    z: Complex,
    x: f64,
    y: f64,
) -> [Complex; 4] {
    let mut energy = [Complex::ZERO; 4];
    for (c, (s, t)) in SPINS.into_iter().enumerate() {
        let pairs: Vec<_> = if c == 1 {
            (0..16).map(decoded).collect()
        } else {
            vec![[0, 0, 1, 1]]
        };
        let n = pairs.len();
        let mut operator = BlochMatrix::zeros(3 * n, 3 * n);
        let mut left = vec![Complex::ZERO; 3 * n];
        let mut right = vec![Complex::ZERO; 3 * n];
        let raw: Vec<_> = problem.ovov[c].iter().copied().map(Complex::from).collect();
        for h in 0..3 {
            for (i, &row) in pairs.iter().enumerate() {
                let index = h * n + i;
                let unperturbed = unique_value(&raw, c, row);
                let amplitude = |v: usize, scale: f64, left: bool| {
                    let vertex = &vertices[v];
                    let terms = if left {
                        &vertex.coupling_left
                    } else {
                        &vertex.coupling_right
                    };
                    unique_value(&terms[c], c, row) * scale
                };
                left[index] = match h {
                    0 => amplitude(0, x, true),
                    1 => unperturbed + amplitude(2, x * y, true),
                    _ => amplitude(1, y, true),
                };
                right[index] = match h {
                    0 => amplitude(1, y, false),
                    1 => unperturbed + amplitude(2, x * y, false),
                    _ => amplitude(0, x, false),
                };
                for k in 0..3 {
                    for (j, &col) in pairs.iter().enumerate() {
                        let mut value = match h as isize - k as isize {
                            0 => {
                                unique_action(c, &vertices[2].orbital_generator, row, col) * (x * y)
                            }
                            1 => unique_action(c, &vertices[0].orbital_generator, row, col) * x,
                            -1 => unique_action(c, &vertices[1].orbital_generator, row, col) * y,
                            _ => Complex::ZERO,
                        };
                        if h == k && i == j {
                            let [i, a, j, b] = row;
                            value += Complex::from(
                                problem.energies[s][i] + problem.energies[t][j]
                                    - problem.energies[s][2 + a]
                                    - problem.energies[t][2 + b],
                            ) + z * (h as f64 - 1.0);
                        }
                        operator.re[(index, k * n + j)] = value.re;
                        operator.im[(index, k * n + j)] = value.im;
                    }
                }
            }
        }
        let real_rhs: Vec<_> = right
            .iter()
            .map(|v| v.re)
            .chain(right.iter().map(|v| v.im))
            .collect();
        let solution = solve_linear(&operator.real_embedding(), &real_rhs).unwrap();
        energy[c] = left.iter().enumerate().fold(Complex::ZERO, |sum, (i, g)| {
            sum + *g * Complex::new(solution[i], solution[3 * n + i])
        });
    }
    energy[3] =
        (energy[0] + energy[2]) * COEFFICIENTS.same_spin + energy[1] * COEFFICIENTS.opposite_spin;
    energy
}

fn report_values(report: &Pt2FrequencyResponse) -> [Complex; 4] {
    [
        report.channels[0],
        report.channels[1],
        report.channels[2],
        report.correlation,
    ]
}

fn complex_first(plus: Complex, minus: Complex, h: f64) -> Complex {
    Complex::new(
        Stencil::Second.first(plus.re, minus.re, 0.0, 0.0, h),
        Stencil::Second.first(plus.im, minus.im, 0.0, 0.0, h),
    )
}

#[test]
fn pt2_frequency_quadratic_matches_dense_floquet_all_modes_and_channels() {
    let problem = problem();
    let kernel = Pt2FrequencyKernel::new(&problem, COEFFICIENTS).unwrap();
    let z = Complex::new(0.51, 0.039);
    let pairs: Vec<_> = (0..MODES)
        .flat_map(|a| (a..MODES).map(move |b| (a, b)))
        .collect();
    let names: Vec<_> = pairs
        .iter()
        .flat_map(|&(a, b)| {
            ["SS-alpha", "OS", "SS-beta", "total"].map(|c| format!("PT2 Floquet {c} [{a},{b}]"))
        })
        .collect();
    let fixtures: Vec<_> = pairs
        .iter()
        .map(|&(a, b)| {
            let [plus, minus, mixed] = vertices(a, b, z);
            let report = kernel.quadratic(z, [&plus, &minus], &mixed).unwrap();
            let neg = Complex::new(-z.re, z.im);
            let [p, m, pm] = vertices(a, b, neg);
            let negative = kernel.quadratic(neg, [&p, &m], &pm).unwrap();
            let expected = report_values(&report);
            assert!(expected
                .iter()
                .zip(report_values(&negative))
                .all(|(a, b)| (a.conj() - b).abs() < 1e-15));
            assert!(report
                .left_relative_residuals
                .iter()
                .flatten()
                .chain(report.right_relative_residuals.iter().flatten())
                .all(|&r| r < 4e-16));
            ([plus, minus, mixed], expected)
        })
        .collect();
    let steps = [0.10, 0.075, 0.050, 0.030];
    let mut errors = vec![[0.0; 4]; names.len()];
    for (p, (vertices, expected)) in fixtures.iter().enumerate() {
        for (n, &h) in steps.iter().enumerate() {
            let pp = dense_floquet_energy(&problem, vertices, z, h, h);
            let pm = dense_floquet_energy(&problem, vertices, z, h, -h);
            let mp = dense_floquet_energy(&problem, vertices, z, -h, h);
            let mm = dense_floquet_energy(&problem, vertices, z, -h, -h);
            for c in 0..4 {
                let numerical = -complex_first(
                    complex_first(pp[c], pm[c], h),
                    complex_first(mp[c], mm[c], h),
                    h,
                );
                errors[4 * p + c][n] = (expected[c] - numerical).abs();
            }
        }
    }
    // Fit the absolute error, not its square. Zero harmonic requires equal
    // powers of the independent plus/minus amplitudes: E = E0 + a*x*y
    // + b*x^2*y^2 + c*x^3*y^3 + ... . Four-corner parity removes the b term;
    // dividing the first surviving omitted term by h^2 leaves order h^4.
    // Ordinary central differences (including the assembled cosine-field
    // molecular oracle) retain order h^2 and are fitted against order two.
    let failures: Vec<_> = names
        .into_iter()
        .zip(errors)
        .filter_map(|(name, errors)| {
            (!components::component_fit_of_order(&name, &steps, &errors, 4.0)).then_some(name)
        })
        .collect();
    components::assert_no_failures("analytic PT2 Floquet mixed derivatives", &failures);
    println!("Damped PT2 Floquet response [0,0] {:?}", fixtures[0].1);
}

#[test]
fn pt2_frequency_zero_recovers_every_static_pair_hessian_channel() {
    let problem = problem();
    let kernel = Pt2FrequencyKernel::new(&problem, COEFFICIENTS).unwrap();
    let block = |vertex: &Pt2FrequencyVertex| {
        vertex
            .coupling_right
            .each_ref()
            .map(|v| v.iter().map(|z| z.re).collect::<Vec<_>>())
    };
    let zero = std::array::from_fn(|c| vec![0.0; problem.ovov[c].len()]);
    let zero_fock = std::array::from_fn(|_| Matrix::zeros(4, 4));
    let fock =
        |vertex: &Pt2FrequencyVertex| vertex.orbital_generator.each_ref().map(|f| f.re.clone());
    let hessian = |a: &Pt2FrequencyVertex, b: &Pt2FrequencyVertex, mixed: &Pt2FrequencyVertex| {
        let x = block(a);
        let y = block(b);
        let sum = std::array::from_fn(|c| x[c].iter().zip(&y[c]).map(|(x, y)| x + y).collect());
        let fx = fock(a);
        let fy = fock(b);
        let sf = std::array::from_fn(|s| {
            Matrix::from_row_major(
                4,
                4,
                fx[s]
                    .as_slice()
                    .iter()
                    .zip(fy[s].as_slice())
                    .map(|(x, y)| x + y)
                    .collect(),
            )
        });
        let energy = |coupling: &[[Vec<f64>; 3]], operator: &[[Matrix; 2]]| {
            let solution = taylor_response(&problem, coupling, operator, COEFFICIENTS).unwrap();
            [
                solution.same_spin_alpha[2],
                solution.opposite_spin[2],
                solution.same_spin_beta[2],
                solution.correlation[2],
            ]
        };
        let combined = energy(&[sum, block(mixed)], &[sf, fock(mixed)]);
        let first = energy(&[x, zero.clone()], &[fx, zero_fock.clone()]);
        let second = energy(&[y, zero.clone()], &[fy, zero_fock.clone()]);
        std::array::from_fn::<_, 4, _>(|c| -combined[c] + first[c] + second[c])
    };
    let mut defect = 0.0_f64;
    for a in 0..MODES {
        for b in 0..MODES {
            let [x, y, mixed] = vertices(a, b, Complex::ZERO);
            let value = kernel.quadratic(Complex::ZERO, [&x, &y], &mixed).unwrap();
            let expected = hessian(&x, &y, &mixed);
            for (actual, exact) in report_values(&value).into_iter().zip(expected) {
                defect = defect.max((actual - Complex::from(exact)).abs());
            }
        }
    }
    println!("Zero-frequency PT2 all-channel static defect {defect:.17e}");
    assert!(defect < 4e-16);
}

#[test]
fn pt2_frequency_moving_frame_phase_covariance() {
    let problem = problem();
    let kernel = Pt2FrequencyKernel::new(&problem, COEFFICIENTS).unwrap();
    let z = Complex::new(0.51, 0.039);
    let mut defect = 0.0_f64;
    for a in 0..MODES {
        for b in 0..MODES {
            let [mut plus, mut minus, mut mixed] = vertices(a, b, z);
            let before = kernel.quadratic(z, [&plus, &minus], &mixed).unwrap();
            let phase = |h: usize, spin: usize, orbital: usize| {
                Complex::new(0.0, 0.17 * (1 + h + 2 * spin + 3 * orbital) as f64)
            };
            let [p, m] = [a, b + MODES];
            for (c, (s, t)) in SPINS.into_iter().enumerate() {
                for k in 0..problem.ovov[c].len() {
                    let [i, a, j, b] = decoded(k);
                    let pair_phase = |h| {
                        phase(h, s, i) + phase(h, t, j) - phase(h, s, 2 + a) - phase(h, t, 2 + b)
                    };
                    let x = pair_phase(p);
                    let y = pair_phase(m);
                    let base = problem.ovov[c][k];
                    // W=exp(x lambda_+ + y lambda_-) is the pair frame.
                    // The left coupling transforms contragrediently.
                    mixed.coupling_right[c][k] += x * minus.coupling_right[c][k]
                        + y * plus.coupling_right[c][k]
                        + x * y * base;
                    mixed.coupling_left[c][k] += -x * minus.coupling_left[c][k]
                        - y * plus.coupling_left[c][k]
                        + x * y * base;
                    plus.coupling_right[c][k] += x * base;
                    minus.coupling_right[c][k] += y * base;
                    plus.coupling_left[c][k] -= x * base;
                    minus.coupling_left[c][k] -= y * base;
                }
            }
            for s in 0..2 {
                for i in 0..4 {
                    for j in 0..4 {
                        // U^{-1} F U contributes [F_+,K_-]+[F_-,K_+]
                        // at the mixed harmonic for diagonal generators.
                        let change = matrix_element(&plus.orbital_generator[s], i, j)
                            * (phase(m, s, j) - phase(m, s, i))
                            + matrix_element(&minus.orbital_generator[s], i, j)
                                * (phase(p, s, j) - phase(p, s, i));
                        mixed.orbital_generator[s].re[(i, j)] += change.re;
                        mixed.orbital_generator[s].im[(i, j)] += change.im;
                    }
                    // The Berry connection is the time derivative of the phase.
                    for (vertex, connection) in [
                        (&mut plus, -z * phase(p, s, i)),
                        (&mut minus, z * phase(m, s, i)),
                    ] {
                        vertex.orbital_generator[s].re[(i, i)] += connection.re;
                        vertex.orbital_generator[s].im[(i, i)] += connection.im;
                    }
                }
            }
            let after = kernel.quadratic(z, [&plus, &minus], &mixed).unwrap();
            for (a, b) in report_values(&before)
                .into_iter()
                .zip(report_values(&after))
            {
                defect = defect.max((a - b).abs());
            }
        }
    }
    println!("Moving-frame phase covariance of PT2 response {defect:.17e}");
    assert!(defect < 4e-15);
}

#[test]
fn pt2_frequency_excludes_pauli_forbidden_poles() {
    let clean = problem();
    let mut perturbed = clean.clone();
    let forbidden = [1, 0, 1, 1];
    let allowed = [0, 0, 1, 1];
    let [i, a, j, b] = forbidden;
    // The raw Coulomb tensor may have a nonzero direct entry here, but
    // exchange cancels it. A one-ULP discrepancy must not create a state.
    perturbed.ovov[0][packed(forbidden)] = 0.125;
    perturbed.ovov[0][packed([i, b, j, a])] = f64::from_bits(0.125_f64.to_bits() + 1);
    let pole = |[i, a, j, b]: [usize; 4]| {
        let e = &clean.energies[0];
        -(e[i] + e[j] - e[2 + a] - e[2 + b])
    };
    let response = |problem: &Pt2Problem<f64>, z: Complex| {
        let kernel = Pt2FrequencyKernel::new(problem, COEFFICIENTS).unwrap();
        let [plus, minus, mixed] = vertices(0, 1, z);
        kernel.quadratic(z, [&plus, &minus], &mixed)
    };
    let z = Complex::from(pole(forbidden));
    let reference = response(&clean, z).unwrap();
    let measured = response(&perturbed, z).unwrap();
    assert_eq!(report_values(&measured), report_values(&reference));
    assert!(measured
        .minimum_driven_denominator
        .is_some_and(|gap| gap > 0.0));
    let real_pole = pole(allowed);
    assert!(matches!(
        response(&clean, Complex::from(real_pole)),
        Err(DftError::Unsupported { feature, .. }) if feature.contains("driven pair pole")
    ));
    let damped = response(&clean, Complex::new(real_pole, 0.01)).unwrap();
    assert!(damped.correlation.is_finite());
}

#[test]
fn frequency_dual_ket_subset_transform_fits_all_components() {
    const N: usize = 3;
    let energies = [vec![-1.1, 0.3, 0.6], vec![-0.9, -0.4, 0.8]];
    let occupied = [1, 2];
    let tensor: Vec<_> = (0..N.pow(4))
        .map(|k| 0.1 + 0.07 * (k as f64 + 0.31).sin())
        .collect();
    let paths = [0, 1].map(|frame| {
        std::array::from_fn::<_, 2, _>(|spin| {
            (0..4)
                .map(|lane| {
                    let mut c = ComplexMatrix::zeros(N, N);
                    for a in 0..N {
                        for i in 0..N {
                            let seed = (1 + 5 * frame + 2 * spin + 3 * lane + 2 * a + 4 * i) as f64;
                            let base = if lane == 0 && a == i { 0.8 } else { 0.0 };
                            c.re[(a, i)] = base + 0.24 * (0.37 * seed).sin();
                            c.im[(a, i)] = 0.13 * (0.61 * seed).cos();
                        }
                    }
                    c
                })
                .collect::<Vec<_>>()
        })
    });
    let analytic = super::super::transform_frequency_path(
        N, &tensor, &paths[0], &paths[1], &energies, occupied,
    )
    .unwrap();
    let numerical = |x: f64, y: f64| {
        let value = |frame: usize, spin: usize, row: usize, col: usize| {
            let coefficients = &paths[frame][spin];
            [1.0, x, y, x * y]
                .into_iter()
                .zip(coefficients)
                .fold(Complex::ZERO, |sum, (scale, c)| {
                    sum + matrix_element(c, row, col) * scale
                })
        };
        std::array::from_fn::<_, 3, _>(|c| {
            let (s, t) = SPINS[c];
            let mut output = Vec::new();
            for i in 0..occupied[s] {
                for a in occupied[s]..N {
                    for j in 0..occupied[t] {
                        for b in occupied[t]..N {
                            let mut term = Complex::ZERO;
                            for p in 0..N {
                                for q in 0..N {
                                    for r in 0..N {
                                        for u in 0..N {
                                            let integral = tensor[((p * N + q) * N + r) * N + u];
                                            term += value(0, s, p, i)
                                                * value(1, s, q, a)
                                                * value(0, t, r, j)
                                                * value(1, t, u, b)
                                                * integral;
                                        }
                                    }
                                }
                            }
                            output.push(term);
                        }
                    }
                }
            }
            output
        })
    };
    let reference = numerical(0.0, 0.0);
    for c in 0..3 {
        for (got, expected) in analytic[0][c].iter().zip(&reference[c]) {
            assert!((*got - *expected).abs() < 3e-15);
        }
    }
    let names: Vec<_> = (1..4)
        .flat_map(|lane| {
            let expected = &analytic[lane];
            (0..3).flat_map(move |c| {
                (0..expected[c].len())
                    .map(move |p| format!("Fourier dual/ket lane {lane} spin {c} element {p}"))
            })
        })
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.10, 0.075, 0.050, 0.030], [0.05, 0.035, 0.025, 0.016]],
        |steps| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (step, &h) in steps.iter().enumerate() {
                let x_p = numerical(h, 0.0);
                let x_m = numerical(-h, 0.0);
                let y_p = numerical(0.0, h);
                let y_m = numerical(0.0, -h);
                let pp = numerical(h, h);
                let pm = numerical(h, -h);
                let mp = numerical(-h, h);
                let mm = numerical(-h, -h);
                let mut k = 0;
                for lane in 1..4 {
                    for c in 0..3 {
                        for p in 0..analytic[lane][c].len() {
                            let measured = match lane {
                                1 => complex_first(x_p[c][p], x_m[c][p], h),
                                2 => complex_first(y_p[c][p], y_m[c][p], h),
                                _ => complex_first(
                                    complex_first(pp[c][p], pm[c][p], h),
                                    complex_first(mp[c][p], mm[c][p], h),
                                    h,
                                ),
                            };
                            errors[k][step] = (analytic[lane][c][p] - measured).abs();
                            k += 1;
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("Fourier dual/ket integral transform", &failures);
}
