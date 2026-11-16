// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full complex AO-density derivatives of the exact local LDA spin limit.

#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;

use dft_rs::periodic::spinor_scf::from_pauli;
use dft_rs::periodic::spinor_xc::spinor_semilocal_grid;
use dft_rs::periodic::BlochMatrix;
use dft_rs::xc::{resolve, Family, FunctionalSpec, GridVars, ResolvedFunctional, XcOut};

fn fixture() -> (BlochMatrix, [BlochMatrix; 3], [BlochMatrix; 4]) {
    let mut phi = BlochMatrix::zeros(3, 2);
    for (g, row) in [
        [(0.91, 0.31), (0.47, -0.69)],
        [(0.38, -0.42), (1.12, 0.20)],
        [(0.68, 0.73), (-0.39, 0.84)],
    ]
    .into_iter()
    .enumerate()
    {
        for (a, (re, im)) in row.into_iter().enumerate() {
            phi.re[(g, a)] = re;
            phi.im[(g, a)] = im;
        }
    }
    let mut momentum = std::array::from_fn(|_| BlochMatrix::zeros(3, 2));
    for (d, matrix) in momentum.iter_mut().enumerate() {
        for g in 0..3 {
            for a in 0..2 {
                matrix.re[(g, a)] = 0.11 * ((d + 1) * (a + 2) + g) as f64;
                matrix.im[(g, a)] = -0.07 * ((d + 2) * (g + 1) + a) as f64;
            }
        }
    }
    let mut p = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    for (c, (d0, d1, re, im)) in [
        (0.95, 0.83, 0.07, -0.03),
        (0.24, -0.15, 0.04, 0.02),
        (-0.13, 0.19, -0.03, 0.05),
        (0.17, 0.11, 0.06, -0.04),
    ]
    .into_iter()
    .enumerate()
    {
        p[c].re[(0, 0)] = d0;
        p[c].re[(1, 1)] = d1;
        p[c].re[(0, 1)] = re;
        p[c].re[(1, 0)] = re;
        p[c].im[(0, 1)] = im;
        p[c].im[(1, 0)] = -im;
    }
    (phi, momentum, p)
}

// The scalar oracle diagonalizes each local 2x2 spin-density matrix through
// rho +/- |m|. It does not call spin projection, its gradient or AO assembly.
fn local_lda_energy(
    f: &ResolvedFunctional,
    weights: &[f64],
    phi: &BlochMatrix,
    p: &[BlochMatrix; 4],
) -> f64 {
    assert_eq!(f.family, Family::Lda);
    let mut energy = 0.0;
    for (g, weight) in weights.iter().enumerate() {
        let mut x = [0.0; 4];
        for c in 0..4 {
            for a in 0..2 {
                for b in 0..2 {
                    let re = phi.re[(g, a)] * phi.re[(g, b)] + phi.im[(g, a)] * phi.im[(g, b)];
                    let im = phi.im[(g, a)] * phi.re[(g, b)] - phi.re[(g, a)] * phi.im[(g, b)];
                    x[c] += p[c].re[(a, b)] * re - p[c].im[(a, b)] * im;
                }
            }
        }
        let magnitude = x[1].hypot(x[2]).hypot(x[3]);
        assert!(x[0] > magnitude);
        let up = [(x[0] + magnitude) / 2.0];
        let down = [(x[0] - magnitude) / 2.0];
        let mut out = XcOut::default();
        f.eval(
            &GridVars {
                rho: [&up, &down],
                sigma: None,
                tau: None,
            },
            0,
            &mut out,
        )
        .unwrap();
        energy += weight * out.e[0];
    }
    energy
}

#[test]
fn all_sixteen_hermitian_density_components_fit_independent_local_spin_eigenvalues() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let (phi, momentum, p) = fixture();
            let f = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
            let weights = [0.31, 0.23, 0.47];
            let result = spinor_semilocal_grid(
                &f,
                &weights,
                &[&phi],
                &[&momentum],
                std::slice::from_ref(&p),
                4,
            )
            .unwrap();
            assert!((result.0 - local_lda_energy(&f, &weights, &phi, &p)).abs() < 1e-13);
            assert!(result.1[0].hermiticity_error() < 1e-13);
            let directions: Vec<_> = (0..16)
                .map(|i| {
                    let mut delta = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
                    let m = &mut delta[i / 4];
                    match i % 4 {
                        0 => m.re[(0, 0)] = 0.4,
                        1 => m.re[(1, 1)] = 0.4,
                        2 => {
                            m.re[(0, 1)] = 0.3;
                            m.re[(1, 0)] = 0.3;
                        }
                        _ => {
                            m.im[(0, 1)] = 0.3;
                            m.im[(1, 0)] = -0.3;
                        }
                    }
                    delta
                })
                .collect();
            let analytic: Vec<_> = directions
                .iter()
                .map(|delta| from_pauli(delta).unwrap().trace_product(&result.1[0]))
                .collect();
            let labels: Vec<_> = (0..16)
                .map(|i| format!("local-spin LDA Pauli{} Hermitian{}", i / 4, i % 4))
                .collect();
            let mut failures = Vec::new();
            components::fit_components(
                &labels,
                &[
                    [0.08, 0.04, 0.02, 0.01],
                    [0.04, 0.02, 0.01, 0.005],
                    [0.16, 0.08, 0.04, 0.02],
                ],
                |steps| {
                    directions
                        .iter()
                        .zip(&analytic)
                        .map(|(delta, exact)| {
                            std::array::from_fn(|s| {
                                let h = steps[s];
                                let energy = |sign: f64| {
                                    let varied = std::array::from_fn(|c| {
                                        let mut m = p[c].clone();
                                        m.add_scaled(&delta[c], sign * h);
                                        m
                                    });
                                    local_lda_energy(&f, &weights, &phi, &varied)
                                };
                                ((energy(1.0) - energy(-1.0)) / (2.0 * h) - exact).abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            components::assert_no_failures("full complex local-spin LDA grid", &failures);
            let mut invalid = momentum.clone();
            invalid[0].re[(0, 0)] = f64::NAN;
            assert!(spinor_semilocal_grid(
                &f,
                &weights,
                &[&phi],
                &[&invalid],
                std::slice::from_ref(&p),
                4
            )
            .is_err());
            let mut invalid_density = p;
            invalid_density[0].scale(0.01);
            assert!(spinor_semilocal_grid(
                &f,
                &weights,
                &[&phi],
                &[&momentum],
                &[invalid_density],
                4
            )
            .is_err());
        })
        .unwrap();
}

#[test]
fn gga_ignores_only_inactive_transverse_kinetic_and_current_descriptors() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let (mut phi, mut momentum, mut p) = fixture();
            phi.im.as_mut_slice().fill(0.0);
            for matrix in &mut momentum {
                matrix.im.as_mut_slice().fill(0.0);
            }
            for matrix in &mut p {
                matrix.im.as_mut_slice().fill(0.0);
            }
            // Real values and momenta make every density gradient zero.
            // Distinct spin/AO matrices still give transverse currents.
            let contract = |c: usize, left: &BlochMatrix, right: &BlochMatrix| {
                (0..2)
                    .flat_map(|a| (0..2).map(move |b| (a, b)))
                    .map(|(a, b)| p[c].re[(a, b)] * left.re[(0, a)] * right.re[(0, b)])
                    .sum::<f64>()
            };
            let m: [f64; 3] = std::array::from_fn(|a| contract(a + 1, &phi, &phi));
            let j: [f64; 3] =
                std::array::from_fn(|a| contract(a + 1, &momentum[0], &phi));
            let cross = [
                m[1] * j[2] - m[2] * j[1],
                m[2] * j[0] - m[0] * j[2],
                m[0] * j[1] - m[1] * j[0],
            ];
            assert!(cross.iter().map(|v| v * v).sum::<f64>().sqrt() > 1e-5);
            let weights = [0.31, 0.23, 0.47];
            let f = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
            let result = spinor_semilocal_grid(
                &f,
                &weights,
                &[&phi],
                &[&momentum],
                std::slice::from_ref(&p),
                4,
            )
            .unwrap();
            let mut expected_energy = 0.0;
            let mut expected = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
            for (g, weight) in weights.into_iter().enumerate() {
                let x: [f64; 4] = std::array::from_fn(|c| {
                    (0..2)
                        .flat_map(|a| (0..2).map(move |b| (a, b)))
                        .map(|(a, b)| p[c].re[(a, b)] * phi.re[(g, a)] * phi.re[(g, b)])
                        .sum()
                });
                let magnitude = x[1].hypot(x[2]).hypot(x[3]);
                let rho = [[0.5 * (x[0] + magnitude)], [0.5 * (x[0] - magnitude)]];
                let zero = [0.0];
                let mut out = XcOut::default();
                f.eval(
                    &GridVars {
                        rho: [&rho[0], &rho[1]],
                        sigma: Some([&zero, &zero, &zero]),
                        tau: None,
                    },
                    1,
                    &mut out,
                )
                .unwrap();
                expected_energy += weight * out.e[0];
                let charge = 0.5 * (out.vrho[0][0] + out.vrho[1][0]);
                let spin = 0.5 * (out.vrho[0][0] - out.vrho[1][0]) / magnitude;
                for c in 0..4 {
                    let coefficient = if c == 0 { charge } else { spin * x[c] };
                    for a in 0..2 {
                        for b in 0..2 {
                            expected[c].re[(a, b)] +=
                                2.0 * weight * coefficient * phi.re[(g, a)] * phi.re[(g, b)];
                        }
                    }
                }
            }
            let expected_fock = from_pauli(&expected).unwrap();
            assert!((result.0 - expected_energy).abs() < 1e-13);
            assert!(result.1[0].re.max_abs_diff(&expected_fock.re) < 1e-12);
            assert!(result.1[0].im.max_abs_diff(&expected_fock.im) < 1e-12);
            println!("GGA inactive transverse current: all complex Fock entries match the independent zero-gradient scalar spin-eigenvalue limit");
        })
        .unwrap();
}
