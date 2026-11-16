// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All independent complex pair derivatives on an odd-electron spinor
//! space, against independently assembled dense physical pair equations.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::integrals::london::ComplexMatrix as Cm;
use dft_rs::linalg::{solve_linear, Matrix};
use dft_rs::properties::pt2_response::{generalized_taylor_response, spinor_transform_path};
use dft_rs::response::complex::Complex as C;

const NO: usize = 3;
const NV: usize = 2;
const NM: usize = NO + NV;
const NP: usize = NO * NV * NO * NV;
fn ix(i: usize, a: usize, j: usize, b: usize) -> usize {
    ((i * NV + a) * NO + j) * NV + b
}
fn indices(mut p: usize) -> [usize; 4] {
    let b = p % NV;
    p /= NV;
    let j = p % NO;
    p /= NO;
    let a = p % NV;
    [p / NV, a, j, b]
}
fn z(m: &Cm, i: usize, j: usize) -> C {
    C::new(m.re[(i, j)], m.im[(i, j)])
}

#[test]
fn odd_electron_complex_base_has_complete_dense_pair_fits() {
    let energies = [-1.4, -1.0, -0.6, 0.5, 0.9];
    let raw = |p: usize, k: usize| {
        let [i, a, j, b] = indices(p);
        let q = ix(j, b, i, a);
        let v = (p.min(q) + 13 * k + 1) as f64;
        C::new(0.13 * (0.43 * v).sin(), 0.07 * (0.37 * v).cos())
    };
    let sources = (0..=2)
        .map(|k| (0..NP).map(|p| raw(p, k)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let fock = (0..=2)
        .map(|k| {
            let mut f = Cm::zeros(NM, NM);
            for i in 0..NM {
                for j in 0..NM {
                    if k == 0 {
                        if i == j {
                            f.re[(i, j)] = energies[i];
                        }
                    } else if (i < NO) == (j < NO) {
                        f.re[(i, j)] = 0.12 * ((i + j + 5 * k + 1) as f64 * 0.31).cos();
                        f.im[(i, j)] =
                            0.09 * (i as f64 - j as f64) * ((i + j + k + 1) as f64 * 0.27).sin();
                    }
                }
            }
            f
        })
        .collect::<Vec<_>>();
    let response = generalized_taylor_response(&energies, NO, &sources, &fock).unwrap();
    assert!(response.residuals.iter().all(|v| *v < 1e-13));
    assert!(response.energy.iter().all(|v| v.im.abs() < 1e-14));
    let at = |t: f64| {
        let element =
            |i, j| z(&fock[0], i, j) + z(&fock[1], i, j) * t + z(&fock[2], i, j) * (t * t);
        let source = (0..NP)
            .map(|p| {
                let [i, a, j, b] = indices(p);
                let q = ix(i, b, j, a);
                (0..=2).fold(C::ZERO, |g, k| {
                    g + (sources[k][p] - sources[k][q]) * t.powi(k as i32)
                })
            })
            .collect::<Vec<_>>();
        let mut a = Matrix::zeros(2 * NP, 2 * NP);
        for p in 0..NP {
            let [i, av, j, b] = indices(p);
            for q in 0..NP {
                let [k, c, l, d] = indices(q);
                let mut v = C::ZERO;
                if av == c && j == l && b == d {
                    v += element(k, i);
                }
                if i == k && av == c && b == d {
                    v += element(l, j);
                }
                if i == k && j == l && b == d {
                    v -= element(NO + av, NO + c);
                }
                if i == k && av == c && j == l {
                    v -= element(NO + b, NO + d);
                }
                a[(p, q)] = v.re;
                a[(p, q + NP)] = -v.im;
                a[(p + NP, q)] = v.im;
                a[(p + NP, q + NP)] = v.re;
            }
        }
        let rhs = source
            .iter()
            .map(|g| g.re)
            .chain(source.iter().map(|g| g.im))
            .collect::<Vec<_>>();
        let x = solve_linear(&a, &rhs).unwrap();
        let amp = (0..NP).map(|p| C::new(x[p], x[p + NP])).collect::<Vec<_>>();
        let e = source
            .iter()
            .zip(&amp)
            .fold(C::ZERO, |e, (g, t)| e + g.conj() * *t)
            * 0.25;
        assert!(e.im.abs() < 1e-13);
        (amp, e.re)
    };
    let centre = at(0.0);
    for p in 0..NP {
        assert!((centre.0[p] - response.amplitudes[0][p]).abs() < 1e-14);
    }
    let locations = (0..NO)
        .flat_map(|i| {
            (i + 1..NO).flat_map(move |j| {
                (0..NV).flat_map(move |a| {
                    (a + 1..NV).flat_map(move |b| [(ix(i, a, j, b), 0), (ix(i, a, j, b), 1)])
                })
            })
        })
        .chain([(NP, 0)])
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    for order in [1, 2] {
        let names = locations
            .iter()
            .map(|(p, part)| format!("generalized PT2 d{order} element={p} part={part}"))
            .collect::<Vec<_>>();
        components::fit_components(
            &names,
            &[
                [0.08, 0.06, 0.04, 0.03],
                [0.02, 0.015, 0.01, 0.0075],
                [0.2, 0.15, 0.1, 0.075],
            ],
            |steps| {
                let mut errors = vec![[0.0; 4]; locations.len()];
                for (s, h) in steps.iter().enumerate() {
                    let plus = at(*h);
                    let minus = at(-*h);
                    for (n, &(p, part)) in locations.iter().enumerate() {
                        let v = |sample: &(Vec<C>, f64)| {
                            if p == NP {
                                sample.1
                            } else if part == 0 {
                                sample.0[p].re
                            } else {
                                sample.0[p].im
                            }
                        };
                        let exact = if p == NP {
                            response.energy[order].re
                        } else if part == 0 {
                            response.amplitudes[order][p].re
                        } else {
                            response.amplitudes[order][p].im
                        };
                        let measured = if order == 1 {
                            (v(&plus) - v(&minus)) / (2.0 * h)
                        } else {
                            (v(&plus) - 2.0 * v(&centre) + v(&minus)) / (2.0 * h * h)
                        };
                        errors[n][s] = (measured - exact).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("complete generalized pair derivatives", &failures);
}

#[test]
fn spinor_transform_retains_every_spin_trace_and_complex_base_element() {
    let nao = 2;
    let mut c = Cm::zeros(4, 4);
    // Dense unitary Fourier spinors: every orbital has both spin components.
    for mu in 0..4 {
        for p in 0..4 {
            let phase = std::f64::consts::FRAC_PI_2 * (mu * p) as f64 + 0.17 * p as f64;
            c.re[(mu, p)] = 0.5 * phase.cos();
            c.im[(mu, p)] = 0.5 * phase.sin();
        }
    }
    let ao = (0..16)
        .map(|p| {
            let mu = p / 8;
            let nu = p / 4 % 2;
            let la = p / 2 % 2;
            let ka = p % 2;
            C::from(0.3 / (1 + mu + nu + la + ka) as f64)
        })
        .collect::<Vec<_>>();
    let source = spinor_transform_path(
        nao,
        std::slice::from_ref(&ao),
        std::slice::from_ref(&c),
        &[-1.1, -0.8, -0.4, 0.7],
        3,
    )
    .unwrap();
    let mut defect = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let a = 3;
            let b = 3;
            let mut expected = C::ZERO;
            for s in 0..2 {
                for t in 0..2 {
                    for mu in 0..nao {
                        for nu in 0..nao {
                            for la in 0..nao {
                                for ka in 0..nao {
                                    expected += z(&c, s * nao + mu, i)
                                        * z(&c, s * nao + nu, a).conj()
                                        * z(&c, t * nao + la, j)
                                        * z(&c, t * nao + ka, b).conj()
                                        * ao[((mu * nao + nu) * nao + la) * nao + ka];
                                }
                            }
                        }
                    }
                }
            }
            defect = defect.max((expected - source[0][i * 3 + j]).abs());
        }
    }
    println!("full spin trace transform max defect={defect:.12e}");
    assert!(defect < 1e-14);
    assert!(source[0].iter().any(|g| g.im.abs() > 1e-5));
}

#[test]
fn generalized_pt2_has_unrestricted_and_global_spin_rotation_limits() {
    use dft_rs::properties::pt2::{from_ao, OrbitalSpace};
    use dft_rs::xc::double_hybrid::Pt2Coefficients;
    let n = 3usize;
    let ao = (0..n.pow(4))
        .map(|p| 0.3 / (1 + p / n.pow(3) + p / n.pow(2) % n + p / n % n + p % n) as f64)
        .collect::<Vec<_>>();
    let energies = [vec![-1.3, -0.7, 0.5], vec![-1.0, 0.3, 0.9]];
    let occupied = [2, 1];
    let collinear = from_ao(
        n,
        &ao,
        &std::array::from_fn(|s| OrbitalSpace {
            energies: energies[s].clone(),
            occupied: occupied[s],
            coefficients: Matrix::identity(n).as_slice().to_vec(),
        }),
    )
    .unwrap()
    .solve(Pt2Coefficients {
        same_spin: 1.0,
        opposite_spin: 1.0,
    })
    .unwrap();
    let labels = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)];
    let eps = labels.map(|(s, i)| energies[s][i]);
    let mut f = Cm::zeros(2 * n, 2 * n);
    for i in 0..2 * n {
        f.re[(i, i)] = eps[i];
    }
    let tensor = ao.into_iter().map(C::from).collect::<Vec<_>>();
    for (theta, phi) in [(0.0_f64, 0.0_f64), (0.37, 0.83), (-0.53, 1.17)] {
        let mut c = Cm::zeros(2 * n, 2 * n);
        let up = C::new(phi.cos(), phi.sin()) * theta.sin();
        for (p, &(s, i)) in labels.iter().enumerate() {
            let pair = if s == 0 {
                [C::from(theta.cos()), up]
            } else {
                [-up.conj(), C::from(theta.cos())]
            };
            for t in 0..2 {
                c.re[(t * n + i, p)] = pair[t].re;
                c.im[(t * n + i, p)] = pair[t].im;
            }
        }
        let source =
            spinor_transform_path(n, std::slice::from_ref(&tensor), &[c], &eps, 3).unwrap();
        let response =
            generalized_taylor_response(&eps, 3, &source, std::slice::from_ref(&f)).unwrap();
        let defect = (response.energy[0] - C::from(collinear.correlation)).abs();
        println!("generalized PT2 collinear/SU2 theta={theta} phi={phi} defect={defect:.12e}");
        assert!(defect < 1e-14);
    }
    assert!(collinear.correlation < 0.0);
}

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/spinor_pair_fixture.rs"]
mod spinor_pair_fixture;

#[test]
fn c1_open_shell_transverse_parent_and_pt2_full_field_hessians() {
    spinor_pair_fixture::fit(
        "C1 three-electron transverse field Hessian",
        |_, basis| {
            let n = basis.nao_sph;
            (0..3)
                .map(|axis| {
                    let mut h = dft_rs::periodic::BlochMatrix::zeros(2 * n, 2 * n);
                    for i in 0..2 * n {
                        for j in i..2 * n {
                            let phase = (i + 3 * j + 7 * axis + 1) as f64;
                            h.re[(i, j)] = 0.04 * (phase * 0.37).sin();
                            h.re[(j, i)] = h.re[(i, j)];
                            if i != j {
                                h.im[(i, j)] = 0.03 * (phase * 0.29).cos();
                                h.im[(j, i)] = -h.im[(i, j)];
                            }
                        }
                    }
                    h
                })
                .collect()
        },
        &[
            [0.08, 0.06, 0.04, 0.03],
            [0.02, 0.015, 0.01, 0.0075],
            [0.2, 0.15, 0.1, 0.075],
        ],
    );
}
