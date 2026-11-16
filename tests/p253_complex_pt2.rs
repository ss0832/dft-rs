// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent dense complex pair equations for every independent SS/OS
//! amplitude derivative and all spin-channel energy derivatives through order 2.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::integrals::london::ComplexMatrix;
use dft_rs::linalg::solve_linear;
use dft_rs::properties::pt2::Pt2Problem;
use dft_rs::properties::pt2_response::complex_taylor_response;
use dft_rs::response::complex::Complex;
use dft_rs::xc::double_hybrid::Pt2Coefficients;
use dft_rs::Matrix;
fn flat(i: usize, a: usize, j: usize, b: usize) -> usize {
    ((i * 2 + a) * 2 + j) * 2 + b
}
fn indices(mut x: usize) -> [usize; 4] {
    let mut out = [0; 4];
    for k in (0..4).rev() {
        out[k] = x % 2;
        x /= 2;
    }
    out
}
fn get(f: &ComplexMatrix, i: usize, j: usize) -> Complex {
    Complex::new(f.re[(i, j)], f.im[(i, j)])
}
#[test]
fn complex_pair_amplitudes_and_channel_energies_fit_independent_dense_solves() {
    let energies = [vec![-1.3, -0.7, 0.4, 0.9], vec![-1.1, -0.5, 0.6, 1.2]];
    let raw = |channel: usize, p: usize, k: usize| {
        let [i, a, j, b] = indices(p);
        let reversed = flat(j, b, i, a);
        let x = (p.min(reversed) + 7 * channel + 11 * k + 1) as f64;
        Complex::new(
            0.12 * (0.43 * x).sin(),
            if k == 0 { 0.0 } else { 0.09 * (0.37 * x).cos() },
        )
    };
    let problem = Pt2Problem {
        energies: energies.clone(),
        occupied: [2, 2],
        ovov: std::array::from_fn(|s| (0..16).map(|p| raw(s, p, 0).re).collect()),
    };
    let paths: Vec<[Vec<Complex>; 3]> = (1..=2)
        .map(|k| std::array::from_fn(|s| (0..16).map(|p| raw(s, p, k)).collect()))
        .collect();
    let focks: Vec<[ComplexMatrix; 2]> = (0..2)
        .map(|k| {
            std::array::from_fn(|s| {
                let mut f = ComplexMatrix::zeros(4, 4);
                for i in 0..4 {
                    for j in 0..4 {
                        if i / 2 == j / 2 {
                            f.re[(i, j)] = 0.11 * ((i + j + 3 * s + 5 * k + 1) as f64 * 0.31).cos();
                            f.im[(i, j)] = 0.07
                                * (i as f64 - j as f64)
                                * ((i + j + 2 * s + k + 1) as f64 * 0.27).sin();
                        }
                    }
                }
                f
            })
        })
        .collect();
    let coefficients = Pt2Coefficients {
        same_spin: 0.31,
        opposite_spin: 0.47,
    };
    let response = complex_taylor_response(&problem, &paths, &focks, coefficients).unwrap();
    assert!(response.residuals.iter().all(|r| *r < 1e-13));
    println!("complex PT2 coefficient residuals {:?}", response.residuals);
    let finite = |h: f64| {
        let mut amplitudes = Vec::new();
        let mut channel_energies = Vec::new();
        for (channel, (left, right)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let f: Vec<_> = [left, right]
                .iter()
                .map(|&s| {
                    let mut f = ComplexMatrix::zeros(4, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            let value = get(&focks[0][s], i, j) * h
                                + get(&focks[1][s], i, j) * (h * h)
                                + Complex::from(if i == j { energies[s][i] } else { 0.0 });
                            f.re[(i, j)] = value.re;
                            f.im[(i, j)] = value.im;
                        }
                    }
                    f
                })
                .collect();
            let mut source = Vec::new();
            for p in 0..16 {
                let [i, a, j, b] = indices(p);
                let value = |q: usize| {
                    Complex::from(problem.ovov[channel][q])
                        + paths[0][channel][q] * h
                        + paths[1][channel][q] * (h * h)
                };
                source.push(
                    value(p)
                        - if left == right {
                            value(flat(i, b, j, a))
                        } else {
                            Complex::ZERO
                        },
                );
            }
            let mut matrix = Matrix::zeros(32, 32);
            for p in 0..16 {
                let [i, a, j, b] = indices(p);
                for q in 0..16 {
                    let [k, c, l, d] = indices(q);
                    // Direct Kronecker formula, independent of the production
                    // matrix-free four-index action and coefficient recursion.
                    let mut z = Complex::ZERO;
                    if a == c && j == l && b == d {
                        z += get(&f[0], k, i);
                    }
                    if i == k && a == c && b == d {
                        z += get(&f[1], l, j);
                    }
                    if i == k && j == l && b == d {
                        z -= get(&f[0], 2 + a, 2 + c);
                    }
                    if i == k && a == c && j == l {
                        z -= get(&f[1], 2 + b, 2 + d);
                    }
                    matrix[(p, q)] = z.re;
                    matrix[(p, q + 16)] = -z.im;
                    matrix[(p + 16, q)] = z.im;
                    matrix[(p + 16, q + 16)] = z.re;
                }
            }
            let rhs: Vec<_> = source
                .iter()
                .map(|g| g.re)
                .chain(source.iter().map(|g| g.im))
                .collect();
            let solution = solve_linear(&matrix, &rhs).unwrap();
            let amp: Vec<_> = (0..16)
                .map(|p| Complex::new(solution[p], solution[p + 16]))
                .collect();
            let energy = source
                .iter()
                .zip(&amp)
                .fold(Complex::ZERO, |e, (g, t)| e + g.conj() * *t)
                * if left == right { 0.25 } else { 1.0 };
            assert!(energy.im.abs() < 1e-13);
            amplitudes.push(amp);
            channel_energies.push(energy.re);
        }
        (amplitudes, channel_energies)
    };
    let stages = [
        [0.08, 0.06, 0.04, 0.03],
        [0.02, 0.015, 0.01, 0.0075],
        [0.2, 0.15, 0.1, 0.075],
    ];
    let baseline = finite(0.0);
    let mut failures = Vec::new();
    for order in [1usize, 2] {
        let mut locations = Vec::new();
        let mut names = Vec::new();
        for channel in 0..3 {
            for p in 0..16 {
                let [i, a, j, b] = indices(p);
                if channel != 1 && !(i < j && a < b) {
                    continue;
                }
                for part in 0..2 {
                    locations.push((channel, p, part));
                    names.push(format!(
                        "complex PT2 d{order} amp channel={channel} element={p} part={part}"
                    ));
                }
            }
        }
        for channel in 0..3 {
            locations.push((channel, 16, 0));
            names.push(format!("complex PT2 d{order} channel energy={channel}"));
        }
        components::fit_components(
            &names,
            &stages,
            |steps| {
                let mut errors = vec![[0.0; 4]; locations.len()];
                for (k, &h) in steps.iter().enumerate() {
                    let plus = finite(h);
                    let minus = finite(-h);
                    for (a, &(s, p, part)) in locations.iter().enumerate() {
                        let actual = if p == 16 {
                            [
                                response.same_spin_alpha[order],
                                response.opposite_spin[order],
                                response.same_spin_beta[order],
                            ][s]
                                .re
                        } else {
                            let z = response.amplitude_coefficients[order][s][p];
                            if part == 0 {
                                z.re
                            } else {
                                z.im
                            }
                        };
                        let value = |sample: &(Vec<Vec<Complex>>, Vec<f64>)| {
                            if p == 16 {
                                sample.1[s]
                            } else {
                                let z = sample.0[s][p];
                                if part == 0 {
                                    z.re
                                } else {
                                    z.im
                                }
                            }
                        };
                        let estimate = if order == 1 {
                            (value(&plus) - value(&minus)) / (2.0 * h)
                        } else {
                            (value(&plus) - 2.0 * value(&baseline) + value(&minus)) / (2.0 * h * h)
                        };
                        // The second-order production value is factorial normalized.
                        errors[a][k] = (estimate - actual).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("all independent complex PT2 coefficients", &failures);
    // Independent orbital-gauge limit: virtual bra and occupied ket phases
    // must cancel in every energy channel, including at first/second order.
    let theta = [[0.13, -0.17, 0.29, -0.11], [-0.09, 0.21, -0.23, 0.19]];
    let phase_paths: Vec<[Vec<Complex>; 3]> = (1..=2)
        .map(|order| {
            std::array::from_fn(|channel| {
                let (left, right) = [(0, 0), (0, 1), (1, 1)][channel];
                (0..16)
                    .map(|p| {
                        let [i, a, j, b] = indices(p);
                        let phase = theta[left][i] + theta[right][j]
                            - theta[left][2 + a]
                            - theta[right][2 + b];
                        if order == 1 {
                            Complex::new(0.0, phase * problem.ovov[channel][p])
                        } else {
                            Complex::from(-0.5 * phase * phase * problem.ovov[channel][p])
                        }
                    })
                    .collect()
            })
        })
        .collect();
    let phase_focks: Vec<[ComplexMatrix; 2]> = (0..2)
        .map(|_| std::array::from_fn(|_| ComplexMatrix::zeros(4, 4)))
        .collect();
    let phase =
        complex_taylor_response(&problem, &phase_paths, &phase_focks, coefficients).unwrap();
    let phase_defect = phase.same_spin_alpha[1..]
        .iter()
        .chain(&phase.same_spin_beta[1..])
        .chain(&phase.opposite_spin[1..])
        .map(|z| z.abs())
        .fold(0.0_f64, f64::max);
    println!("complex PT2 orbital phase energy defect {phase_defect:.12e}");
    assert!(phase_defect < 1e-14);
}

#[test]
fn complex_integral_transform_fits_every_source_component() {
    use dft_rs::properties::pt2_response::complex_transform_path;
    let n = 4usize;
    let energies = [vec![-1.3, -0.7, 0.4, 0.9], vec![-1.1, -0.5, 0.6, 1.2]];
    let integrals: Vec<Vec<Complex>> = (0..3)
        .map(|k| {
            (0..n.pow(4))
                .map(|i| {
                    Complex::new(
                        0.13 * ((i + 7 * k + 1) as f64 * 0.21).sin(),
                        if k == 0 {
                            0.0
                        } else {
                            0.08 * ((2 * i + 3 * k + 1) as f64 * 0.17).cos()
                        },
                    )
                })
                .collect()
        })
        .collect();
    let orbitals: [Vec<ComplexMatrix>; 2] = std::array::from_fn(|s| {
        (0..3)
            .map(|k| {
                let mut c = ComplexMatrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        c.re[(i, j)] = if k == 0 {
                            if i == j {
                                1.0
                            } else {
                                0.0
                            }
                        } else {
                            0.07 * ((i + 3 * j + 5 * k + 2 * s + 1) as f64 * 0.37).sin()
                        };
                        c.im[(i, j)] = if k == 0 {
                            0.0
                        } else {
                            0.05 * ((2 * i + j + 2 * k + 3 * s + 1) as f64 * 0.29).cos()
                        };
                    }
                }
                c
            })
            .collect()
    });
    let transformed = complex_transform_path(n, &integrals, &orbitals, &energies, [2, 2]).unwrap();
    let finite = |h: f64| {
        let coefficients: [ComplexMatrix; 2] = std::array::from_fn(|s| {
            let mut c = ComplexMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let value = get(&orbitals[s][0], i, j)
                        + get(&orbitals[s][1], i, j) * h
                        + get(&orbitals[s][2], i, j) * (h * h);
                    c.re[(i, j)] = value.re;
                    c.im[(i, j)] = value.im;
                }
            }
            c
        });
        [(0, 0), (0, 1), (1, 1)]
            .iter()
            .map(|&(left, right)| {
                (0..16)
                    .map(|p| {
                        let [i, a, j, b] = indices(p);
                        let mut sum = Complex::ZERO;
                        for m in 0..n {
                            for u in 0..n {
                                for v in 0..n {
                                    for w in 0..n {
                                        let k = ((m * n + u) * n + v) * n + w;
                                        let integral = integrals[0][k]
                                            + integrals[1][k] * h
                                            + integrals[2][k] * (h * h);
                                        sum += integral.conj()
                                            * get(&coefficients[left], m, i)
                                            * get(&coefficients[left], u, 2 + a).conj()
                                            * get(&coefficients[right], v, j)
                                            * get(&coefficients[right], w, 2 + b).conj();
                                    }
                                }
                            }
                        }
                        sum
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let baseline = finite(0.0);
    let stages = [
        [0.16, 0.12, 0.08, 0.06],
        [0.04, 0.03, 0.02, 0.015],
        [0.32, 0.24, 0.16, 0.12],
    ];
    let mut failures = Vec::new();
    for order in [1usize, 2] {
        let locations: Vec<_> = (0..3)
            .flat_map(|s| (0..16).flat_map(move |p| (0..2).map(move |part| (s, p, part))))
            .collect();
        let names: Vec<_> = locations
            .iter()
            .map(|(s, p, part)| format!("complex AO d{order} source={s}/{p} part={part}"))
            .collect();
        components::fit_components(
            &names,
            &stages,
            |steps| {
                let mut errors = vec![[0.0; 4]; locations.len()];
                for (k, &h) in steps.iter().enumerate() {
                    let plus = finite(h);
                    let minus = finite(-h);
                    for (q, &(s, p, part)) in locations.iter().enumerate() {
                        let value = |v: Complex| if part == 0 { v.re } else { v.im };
                        let estimate = if order == 1 {
                            (value(plus[s][p]) - value(minus[s][p])) / (2.0 * h)
                        } else {
                            (value(plus[s][p]) - 2.0 * value(baseline[s][p]) + value(minus[s][p]))
                                / (2.0 * h * h)
                        };
                        errors[q][k] = (estimate - value(transformed[order].ovov[s][p])).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("all complex AO-to-MO coefficients", &failures);
}

fn product(a: &ComplexMatrix, b: &ComplexMatrix) -> ComplexMatrix {
    let n = a.re.rows;
    let mut out = ComplexMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let value = (0..n).fold(Complex::ZERO, |sum, k| sum + get(a, i, k) * get(b, k, j));
            out.re[(i, j)] = value.re;
            out.im[(i, j)] = value.im;
        }
    }
    out
}
fn adjoint(a: &ComplexMatrix) -> ComplexMatrix {
    let mut im = a.im.transpose();
    for v in im.as_mut_slice() {
        *v = -*v;
    }
    ComplexMatrix {
        re: a.re.transpose(),
        im,
    }
}
fn add(a: &ComplexMatrix, b: &ComplexMatrix, factor: f64) -> ComplexMatrix {
    let mut out = a.clone();
    for (a, b) in out.re.as_mut_slice().iter_mut().zip(b.re.as_slice()) {
        *a += factor * b;
    }
    for (a, b) in out.im.as_mut_slice().iter_mut().zip(b.im.as_slice()) {
        *a += factor * b;
    }
    out
}
fn inverse(a: &ComplexMatrix) -> ComplexMatrix {
    let n = a.re.rows;
    let mut matrix = Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            let z = get(a, i, j);
            matrix[(i, j)] = z.re;
            matrix[(i, j + n)] = -z.im;
            matrix[(i + n, j)] = z.im;
            matrix[(i + n, j + n)] = z.re;
        }
    }
    let mut out = ComplexMatrix::zeros(n, n);
    for j in 0..n {
        let mut rhs = vec![0.0; 2 * n];
        rhs[j] = 1.0;
        let x = solve_linear(&matrix, &rhs).unwrap();
        for i in 0..n {
            out.re[(i, j)] = x[i];
            out.im[(i, j)] = x[i + n];
        }
    }
    out
}
#[test]
fn complex_projector_transport_fits_every_independent_frame_coefficient() {
    use dft_rs::properties::pt2_response::complex_projector_frames;
    let n = 4;
    let mut identity = ComplexMatrix::zeros(n, n);
    identity.re = Matrix::identity(n);
    let mut occupied = ComplexMatrix::zeros(n, n);
    occupied.re[(0, 0)] = 1.0;
    occupied.re[(1, 1)] = 1.0;
    let mut generator = ComplexMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            generator.re[(i, j)] = 0.13 * (i as f64 - j as f64) * ((i + j + 1) as f64 * 0.31).sin();
            generator.im[(i, j)] = 0.11 * ((i + 2) * (j + 2)) as f64 / 16.0;
        }
    }
    let c = [
        identity.clone(),
        generator.clone(),
        product(&generator, &generator),
    ];
    let c2 = add(&ComplexMatrix::zeros(n, n), &c[2], 0.5);
    let c = [c[0].clone(), c[1].clone(), c2];
    let projectors: Vec<_> = (0..3)
        .map(|k| {
            let mut out = ComplexMatrix::zeros(n, n);
            for j in 0..=k {
                out = add(
                    &out,
                    &product(&product(&c[j], &occupied), &adjoint(&c[k - j])),
                    1.0,
                );
            }
            out
        })
        .collect();
    let (frames, residual) = complex_projector_frames(&projectors, 2).unwrap();
    println!("complex occupied-frame metric/projector residual {residual:.12e}");
    assert!(residual < 1e-13);
    let finite = |h: f64| {
        let orbitals = product(
            &add(&identity, &generator, 0.5 * h),
            &inverse(&add(&identity, &generator, -0.5 * h)),
        );
        let p = product(&product(&orbitals, &occupied), &adjoint(&orbitals));
        let mut q = p.clone();
        for i in 0..n {
            for j in 2..n {
                q.re[(i, j)] = -q.re[(i, j)] + if i == j { 1.0 } else { 0.0 };
                q.im[(i, j)] = -q.im[(i, j)];
            }
        }
        let gram = product(&adjoint(&q), &q);
        // Independent finite-matrix Newton-Schulz inverse square root,
        // rather than the production Taylor binomial transport.
        let mut normalizer = identity.clone();
        for _ in 0..30 {
            let defect = add(
                &identity,
                &product(&gram, &product(&normalizer, &normalizer)),
                -1.0,
            );
            let next = add(&normalizer, &product(&normalizer, &defect), 0.5);
            let difference = next
                .re
                .max_abs_diff(&normalizer.re)
                .max(next.im.max_abs_diff(&normalizer.im));
            normalizer = next;
            if difference < 2e-16 {
                break;
            }
        }
        let frame = product(&q, &normalizer);
        let check = product(&adjoint(&frame), &frame);
        assert!(
            check.re.max_abs_diff(&identity.re) < 1e-13
                && check.im.max_abs_diff(&identity.im) < 1e-13
        );
        frame
    };
    let stages = [
        [0.08, 0.06, 0.04, 0.03],
        [0.02, 0.015, 0.01, 0.0075],
        [0.2, 0.15, 0.1, 0.075],
    ];
    let mut failures = Vec::new();
    for order in [1usize, 2] {
        let mut locations = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for part in 0..2 {
                    if order == 1 && i / 2 == j / 2 || order == 2 && i == j && part == 1 {
                        assert!(if part == 0 {
                            frames[order].re[(i, j)].abs() < 1e-14
                        } else {
                            frames[order].im[(i, j)].abs() < 1e-14
                        });
                        continue;
                    }
                    locations.push((i, j, part));
                }
            }
        }
        let names: Vec<_> = locations
            .iter()
            .map(|(i, j, part)| format!("complex frame d{order} ({i},{j}) part={part}"))
            .collect();
        components::fit_components(
            &names,
            &stages,
            |steps| {
                let mut errors = vec![[0.0; 4]; locations.len()];
                for (k, &h) in steps.iter().enumerate() {
                    let plus = finite(h);
                    let minus = finite(-h);
                    for (q, &(i, j, part)) in locations.iter().enumerate() {
                        let value = |a: &ComplexMatrix| {
                            if part == 0 {
                                a.re[(i, j)]
                            } else {
                                a.im[(i, j)]
                            }
                        };
                        let estimate = if order == 1 {
                            (value(&plus) - value(&minus)) / (2.0 * h)
                        } else {
                            (value(&plus) - 2.0 * value(&identity) + value(&minus)) / (2.0 * h * h)
                        };
                        errors[q][k] = (estimate - value(&frames[order])).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("all independent complex frame coefficients", &failures);
}
