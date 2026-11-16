// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every real and imaginary AO exchange-kernel entry on an independent
//! (k+q,k) space. The oracle enumerates T, U=D-C and C directly, rather than
//! using the production quartet visitor. Four-step fits use the nonlinear
//! density chart P(t)=t E+(t^3+t^5) W: its tangent is the selected unit AO
//! entry E. This tests the kernel, not a self-consistent physical perturbation.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{BasisSet, ElementBasis, Shell};
use dft_rs::dfpt::exchange::FiniteQExchange;
use dft_rs::integrals::{eri_quartet, CoulombKernel};
use dft_rs::periodic::scf::{fock_at_density, PeriodicScfOptions};
use dft_rs::periodic::{BlochMatrix, Cell, Lattice};
use dft_rs::response::complex::Complex;
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;
use dft_rs::Vec3;

fn fixture() -> Cell {
    let mut basis = BasisSet::new("nonsymmetric exchange validation model");
    let atoms = [
        (1, Vec3::new(0.1, -0.2, 0.3), 0.71),
        (2, Vec3::new(1.2, 0.3, -0.1), 0.93),
        (3, Vec3::new(-0.3, 1.1, 0.2), 1.13),
        (4, Vec3::new(0.4, -0.1, 1.5), 0.82),
    ]
    .into_iter()
    .map(|(z, position, exponent)| {
        basis.insert(ElementBasis {
            z,
            shells: vec![Shell {
                l: 0,
                exps: vec![exponent],
                coefs: vec![1.0],
            }],
        });
        Atom { z, position }
    })
    .collect();
    Cell::new(
        Molecule::new(atoms),
        Lattice::new(
            Vec3::new(6.7, 0.31, 0.13),
            Vec3::new(0.2, 7.2, 0.43),
            Vec3::new(0.11, 0.27, 7.7),
        ),
        &basis,
        7.8,
    )
    .unwrap()
}

fn phase(t: f64) -> Complex {
    Complex::new(t.cos(), t.sin())
}

#[test]
fn shared_memory_budget_accounts_multiple_kernels_and_exact_fallback() {
    use dft_rs::resources::{current_usage, reserve, ResourceLimits, MIB};
    let cell = fixture();
    let points = [Vec3::zero()];
    let q = Vec3::new(0.013, -0.017, 0.019);
    let n = cell.nao();
    let mut density = BlochMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            density.re[(i, j)] = ((i + 2 * j + 1) as f64 * 0.23).sin();
            density.im[(i, j)] = ((2 * i + j + 1) as f64 * 0.17).cos() * 0.1;
        }
    }
    let kernels = [CoulombKernel::Coulomb, CoulombKernel::Erf { omega: 0.37 }];
    let reference: Vec<_> = kernels
        .iter()
        .map(|&kernel| {
            FiniteQExchange::with_cache_budget(&cell, &points, q, 2.0, kernel, 0)
                .unwrap()
                .apply(std::slice::from_ref(&density))
                .unwrap()
        })
        .collect();
    ResourceLimits::new(Some(2), Some(4))
        .unwrap()
        .run(|| {
            let operators: Vec<_> = kernels
                .iter()
                .map(|&kernel| FiniteQExchange::new(&cell, &points, q, 2.0, kernel))
                .collect::<dft_rs::Result<_>>()?;
            let charged = current_usage().unwrap().reserved_bytes;
            assert!(charged > 0 && charged < MIB);
            for (operator, expected) in operators.iter().zip(&reference) {
                let actual = operator.apply(std::slice::from_ref(&density))?;
                for (a, b) in flatten(&actual).iter().zip(flatten(expected)) {
                    assert_eq!(*a, b);
                }
                assert!(operator.cached_integral_storage_bound() > 0);
                assert!(operator.cached_integral_storage_bound() <= charged);
            }
            drop(operators);
            assert_eq!(current_usage().unwrap().reserved_bytes, 0);
            let pressure = reserve(4 * MIB - 7, "periodic cache fallback fixture")?;
            let fallback = FiniteQExchange::new(&cell, &points, q, 2.0, kernels[0])?;
            drop(pressure);
            let actual = fallback.apply(std::slice::from_ref(&density))?;
            assert_eq!(fallback.cached_integral_storage_bound(), 0);
            for (a, b) in flatten(&actual).iter().zip(flatten(&reference[0])) {
                assert_eq!(*a, b);
            }
            Ok(())
        })
        .unwrap();
}

// Independent exhaustive Born-von Karman replicas for this bounded fixture.
fn pair_weight(lattice: &Lattice, periods: [usize; 3], displacement: Vec3) -> f64 {
    let mut minimum = f64::INFINITY;
    let mut count = 0;
    let tolerance = 1e-9 * displacement.norm().max(1.0);
    for i in -4..=4 {
        for j in -4..=4 {
            for k in -4..=4 {
                let distance = (displacement
                    + lattice.translation([
                        i * periods[0] as i64,
                        j * periods[1] as i64,
                        k * periods[2] as i64,
                    ]))
                .norm();
                if distance < minimum - tolerance {
                    minimum = distance;
                    count = 1;
                } else if (distance - minimum).abs() <= tolerance {
                    count += 1;
                }
            }
        }
    }
    if displacement.norm() > minimum + tolerance {
        0.0
    } else {
        1.0 / count as f64
    }
}

fn dense_oracle(
    cell: &Cell,
    points: &[Vec3],
    q: Vec3,
    cutoff: f64,
    kernel: CoulombKernel,
    periods: [usize; 3],
) -> Vec<Vec<Complex>> {
    let basis = cell.basis();
    let n = cell.nao();
    let dimension = points.len() * n * n;
    let mut matrix = vec![vec![Complex::ZERO; dimension]; dimension];
    let spread = basis
        .shells
        .iter()
        .flat_map(|a| {
            basis
                .shells
                .iter()
                .map(move |b| (a.center - b.center).norm())
        })
        .fold(0.0_f64, f64::max);
    let translations = cell.lattice().translations_within(cutoff + spread);
    for &t in &translations {
        for &u in &translations {
            for (mu, home_mu) in basis.shells.iter().enumerate() {
                for (nu, home_nu) in basis.shells.iter().enumerate() {
                    let mut moved_nu = home_nu.clone();
                    moved_nu.center += t;
                    if (home_mu.center - moved_nu.center).norm() > cutoff {
                        continue;
                    }
                    let outer =
                        pair_weight(cell.lattice(), periods, moved_nu.center - home_mu.center);
                    if outer == 0.0 {
                        continue;
                    }
                    for (lambda, home_lambda) in basis.shells.iter().enumerate() {
                        for (sigma, home_sigma) in basis.shells.iter().enumerate() {
                            let inner = pair_weight(
                                cell.lattice(),
                                periods,
                                home_sigma.center + u - home_lambda.center,
                            );
                            if inner == 0.0
                                || (home_sigma.center + u - home_lambda.center).norm() > cutoff
                            {
                                continue;
                            }
                            for &c in &translations {
                                let mut moved_lambda = home_lambda.clone();
                                moved_lambda.center += c;
                                let mut moved_sigma = home_sigma.clone();
                                moved_sigma.center += c + u;
                                if (home_mu.center - moved_lambda.center).norm() > cutoff
                                    || (moved_sigma.center - moved_nu.center).norm() > cutoff
                                {
                                    continue;
                                }
                                let integral = eri_quartet(
                                    home_mu,
                                    &moved_lambda,
                                    &moved_sigma,
                                    &moved_nu,
                                    kernel,
                                )
                                .get(0, 0, 0, 0)
                                    * outer
                                    * inner
                                    / points.len() as f64;
                                for (left, &kl) in points.iter().enumerate() {
                                    for (right, &kr) in points.iter().enumerate() {
                                        matrix[(left * n + mu) * n + nu]
                                            [(right * n + lambda) * n + sigma] +=
                                            phase(kl.dot(t) - kr.dot(u) + q.dot(c)) * integral;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    matrix
}

fn unpack(vector: &[Complex], n: usize) -> Vec<BlochMatrix> {
    vector
        .chunks(n * n)
        .map(|block| {
            let mut matrix = BlochMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    matrix.re[(i, j)] = block[i * n + j].re;
                    matrix.im[(i, j)] = block[i * n + j].im;
                }
            }
            matrix
        })
        .collect()
}

fn flatten(blocks: &[BlochMatrix]) -> Vec<Complex> {
    blocks
        .iter()
        .flat_map(|b| {
            b.re.as_slice()
                .iter()
                .zip(b.im.as_slice())
                .map(|(&re, &im)| Complex::new(re, im))
        })
        .collect()
}

fn apply_dense(matrix: &[Vec<Complex>], vector: &[Complex]) -> Vec<Complex> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .fold(Complex::ZERO, |a, (&b, &c)| a + b * c)
        })
        .collect()
}

#[test]
fn all_finite_q_exchange_entries_match_independent_images_and_four_step_fits() {
    let cell = fixture();
    let b = cell.lattice().reciprocal();
    let offset = Vec3::new(0.023, 0.031, 0.017);
    let points = [offset, offset + b[0] * 0.5];
    let q = Vec3::new(0.137, 0.081, -0.069);
    let cutoff = 5.6;
    let dimension = points.len() * cell.nao() * cell.nao();
    let bend = (0..dimension)
        .map(|i| Complex::new(((3 * i + 2) as f64).sin(), ((5 * i + 1) as f64).cos()))
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    for kernel in [
        CoulombKernel::Coulomb,
        CoulombKernel::Erf { omega: 0.37 },
        CoulombKernel::Erfc { omega: 0.37 },
    ] {
        let map = FiniteQExchange::new(&cell, &points, q, cutoff, kernel).unwrap();
        let dense = dense_oracle(&cell, &points, q, cutoff, kernel, [2, 1, 1]);
        let adjoint_defect = (0..dimension)
            .flat_map(|i| (0..dimension).map(move |j| (i, j)))
            .map(|(i, j)| (dense[i][j] - dense[j][i].conj()).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            adjoint_defect < 2e-12,
            "{kernel:?} adjoint defect {adjoint_defect:e}"
        );
        let mut dense_defect = 0.0_f64;
        for source in 0..2 * dimension {
            let mut direction = vec![Complex::ZERO; dimension];
            direction[source / 2] = if source % 2 == 0 {
                Complex::ONE
            } else {
                Complex::new(0.0, 1.0)
            };
            let exact = flatten(&map.apply(&unpack(&direction, cell.nao())).unwrap());
            let oracle = apply_dense(&dense, &direction);
            for (&a, &b) in exact.iter().zip(&oracle) {
                dense_defect = dense_defect.max((a - b).abs());
            }
            let names = (0..2 * dimension)
                .map(|target| format!("{kernel:?} q-exchange input={source} output={target}"))
                .collect::<Vec<_>>();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                    ([0.16, 0.08, 0.04, 0.02], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    let mut errors = vec![[0.0; 4]; 2 * dimension];
                    for (step, &h) in steps.iter().enumerate() {
                        let at = |t: f64| {
                            apply_dense(
                                &dense,
                                &direction
                                    .iter()
                                    .zip(&bend)
                                    .map(|(&e, &w)| e * t + w * (t.powi(3) + t.powi(5)))
                                    .collect::<Vec<_>>(),
                            )
                        };
                        let plus = at(h);
                        let minus = at(-h);
                        let plus2 = at(2.0 * h);
                        let minus2 = at(-2.0 * h);
                        for target in 0..dimension {
                            errors[2 * target][step] = (stencil.first(
                                plus[target].re,
                                minus[target].re,
                                plus2[target].re,
                                minus2[target].re,
                                h,
                            ) - exact[target].re)
                                .abs();
                            errors[2 * target + 1][step] = (stencil.first(
                                plus[target].im,
                                minus[target].im,
                                plus2[target].im,
                                minus2[target].im,
                                h,
                            ) - exact[target].im)
                                .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
        println!("{kernel:?}: all {} real kernel entries, dense defect={dense_defect:.12e}, adjoint defect={adjoint_defect:.12e}",4*dimension*dimension);
        assert!(dense_defect < 2e-12);
        let tiny = 1e-22;
        let full = flatten(&map.apply(&unpack(&bend, cell.nao())).unwrap());
        let small = flatten(
            &map.apply(&unpack(
                &bend.iter().map(|&v| v * tiny).collect::<Vec<_>>(),
                cell.nao(),
            ))
            .unwrap(),
        );
        let defect = full
            .iter()
            .zip(&small)
            .map(|(&a, &b)| (a - b / tiny).abs())
            .fold(0.0_f64, f64::max);
        assert!(defect < 2e-12, "tiny-density linearity {defect:e}");
    }
    components::assert_no_failures("finite-q exchange complete real-linear kernel", &failures);
}

#[test]
fn gamma_exchange_is_the_same_fock_density_derivative_as_reference_scf() {
    let cell = fixture();
    let points = [Vec3::zero()];
    let n = cell.nao();
    let mut p = BlochMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            p.re[(i, j)] = 0.03 * ((i + j + 1) as f64).sin() + if i == j { 0.21 } else { 0.0 };
        }
    }
    let density = [vec![p.clone()], vec![BlochMatrix::zeros(n, n)]];
    let options = PeriodicScfOptions {
        exchange_cutoff: Some(5.6),
        ..Default::default()
    };
    let hf = ResolvedFunctional::hartree_fock();
    let mut hartree = ResolvedFunctional::hartree_fock();
    hartree.hybrid.cx = 0.0;
    let h = fock_at_density(&cell, &hartree, &points, &density, &options, &points, false).unwrap();
    let f = fock_at_density(&cell, &hf, &points, &density, &options, &points, false).unwrap();
    let k = FiniteQExchange::new(&cell, &points, Vec3::zero(), 5.6, CoulombKernel::Coulomb)
        .unwrap()
        .apply(&[p])
        .unwrap();
    let defect = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| {
            (h[0][0].re[(i, j)] - f[0][0].re[(i, j)] - k[0].re[(i, j)])
                .abs()
                .max(k[0].im[(i, j)].abs())
        })
        .fold(0.0_f64, f64::max);
    println!("Gamma reference exchange density derivative defect={defect:.12e}");
    assert!(defect < 2e-12);
}
